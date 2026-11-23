"""Open-answer judge and cache for clinical GRPO."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import aiohttp


JUDGE_SYSTEM_PROMPT = """You are a strict, consistent evaluator of open-ended medical visual question answering. You do not see the medical image, so treat the reference answer as authoritative and judge only whether the candidate answer conveys the same medically relevant answer.

Score on this integer 1-10 scale:
10: Fully correct, complete, and as precise as the reference; an exact or unambiguously equivalent answer.
9: Correct and equivalent, with only negligible wording or specificity differences.
8: Correct core answer with a small non-critical omission or imprecision.
7: Mostly correct, but has a meaningful minor omission, excess claim, or reduced specificity.
6: More correct than incorrect, but incomplete or partly imprecise.
5: Partially correct; captures roughly half of the essential answer.
4: Some relevant overlap, but misses or contradicts a major part of the answer.
3: Mostly incorrect, with only a small correct element.
2: Incorrect, but topically related or showing minimal relevant understanding.
1: Completely incorrect, contradictory, irrelevant, or no usable answer.

Rules:
- Evaluate semantic correctness, not exact string match, grammar, style, verbosity, or presence of reasoning.
- Do not reward unsupported details. Penalize a detail only when it changes, contradicts, or materially overstates the answer.
- For locations, laterality, anatomy, quantities, measurements, and named abnormalities, treat clinically meaningful mismatches as errors.
- For numerical answers, silently convert units and calculate the difference before scoring. Exact equivalent values can receive 10; a rounding difference up to 5% can receive 9; an error up to 10% can receive at most 8; an error from 10% through 25% can receive at most 7; larger errors require a lower score according to severity. These are upper bounds, not automatic scores.
- A concise answer can receive 10. Do not infer facts absent from both the reference and candidate.
- In the JSON, write the brief comparison reason first and the score second. Keep the reason under 160 characters.
- Return only JSON matching the requested schema."""
JUDGE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "open_answer_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "score": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["reason", "score"],
            "additionalProperties": False,
        },
    },
}


def _judge_user_prompt(question: str, reference: str, candidate: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        "First compare the answers in one brief reason, then assign an integer score from 1 to 10."
    )


def _parse_judgment(content: str) -> tuple[int, str]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Judge returned no JSON object: {content[:300]!r}")
        value = json.loads(content[start:end + 1])
    score = value.get("score")
    reason = str(value.get("reason") or "").strip()
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 10:
        raise ValueError(f"Invalid judge score: {content[:300]!r}")
    if not reason:
        raise ValueError(f"Missing judge reason: {content[:300]!r}")
    return score, reason


class JudgeCache:
    def __init__(self, path: str):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS judgments ("
                "fingerprint TEXT PRIMARY KEY, score INTEGER NOT NULL, reason TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    def get(self, fingerprint: str) -> tuple[int, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT score, reason FROM judgments WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
        return (int(row[0]), str(row[1])) if row else None

    def put(self, fingerprint: str, score: int, reason: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO judgments(fingerprint, score, reason) VALUES (?, ?, ?)",
                (fingerprint, score, reason),
            )


class OpenAnswerJudge:
    """Open-answer reward from the served model, scaled from 1–10 to 0–1."""

    def __init__(self, cache_path):
        self.endpoint = os.environ.get(
            "XRAY_GRPO_JUDGE_ENDPOINT", "http://127.0.0.1:8400/v1/chat/completions")
        self.model = os.environ.get("XRAY_GRPO_JUDGE_MODEL", "Qwen3.8-27B")
        self.api_key = os.environ.get("XRAY_GRPO_JUDGE_API_KEY", "EMPTY")
        self.timeout = float(os.environ.get("XRAY_GRPO_JUDGE_TIMEOUT", "300"))
        self.retries = int(os.environ.get("XRAY_GRPO_JUDGE_RETRIES", "5"))
        self.max_tokens = int(os.environ.get("XRAY_GRPO_JUDGE_MAX_TOKENS", "512"))
        self.concurrency = int(os.environ.get("XRAY_GRPO_JUDGE_CONCURRENCY", "16"))
        self.cache = JudgeCache(cache_path)

    @staticmethod
    def _fingerprint(question: str, reference: str, candidate: str, model: str) -> str:
        payload = json.dumps(
            [
                JUDGE_SYSTEM_PROMPT,
                JUDGE_RESPONSE_SCHEMA,
                question,
                reference,
                candidate,
                model,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def score(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        question: str,
        reference: str,
        candidate: str,
    ) -> float:
        fingerprint = self._fingerprint(question, reference, candidate, self.model)
        cached = self.cache.get(fingerprint)
        if cached is not None:
            return (cached[0] - 1) / 9.0

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _judge_user_prompt(question, reference, candidate)},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": JUDGE_RESPONSE_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                async with semaphore:
                    async with session.post(self.endpoint, headers=headers, json=payload) as response:
                        body = await response.text()
                        if response.status != 200:
                            raise RuntimeError(f"Judge HTTP {response.status}: {body[:500]}")
                decoded = json.loads(body)
                score, reason = _parse_judgment(decoded["choices"][0]["message"]["content"])
                self.cache.put(fingerprint, score, reason)
                return (score - 1) / 9.0
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    await asyncio.sleep(min(2**attempt, 10))
        raise RuntimeError(f"LLM judge failed after {self.retries + 1} attempts: {last_error}")
