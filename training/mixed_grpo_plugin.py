"""Clinical GRPO with masked padding and task-specific answer rewards."""

import asyncio
import os
from pathlib import Path

import aiohttp
import torch
from swift.rewards import orms
from swift.rlhf_trainers import GRPOTrainer
from swift.trainers.trainer_factory import TrainerFactory
from transformers import TrainerCallback

from mixed_protocol import local_score
from numerical_protocol import extract_boxed_answer
from open_answer_judge import OpenAnswerJudge


class ExactEpochGuard(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps != 448:
            raise ValueError(f"Expected 448 GRPO updates, got {state.max_steps}")


class MixedGRPOTrainer(GRPOTrainer):
    def _setup_teacher(self):
        super()._setup_teacher()
        if self._has_teacher_explicit():
            raise ValueError("Clinical GRPO uses the frozen KL reference only")
        self._is_dynamic_self_distillation = False
        self._is_self_distillation = False
        self._has_teacher = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.teacher_model is not None or self._has_teacher or self.teacher_kl_coef != 0:
            raise ValueError("Unexpected distillation teacher")
        if self.ref_model is None or self.chord_sft_iterator is not None:
            raise ValueError("Clinical GRPO requires a frozen reference")
        if (
            self.loss_type != "grpo" or self.num_iterations != 1 or self.num_generations != 8
            or self.advantage_estimator != "grpo" or self.scale_rewards != "group" or self.kl_in_reward
            or self.importance_sampling_level != "token" or self.dynamic_sample or self.dynamic_num_samples
            or self.args.async_generate or self.template.padding_free or self.template.sequence_parallel_size != 1
        ):
            raise ValueError("Unexpected clinical GRPO settings")
        real = sum(not row.get("_native_opd_padding", False) for row in self.train_dataset)
        if (
            real != 5371 or len(self.train_dataset) != 5376 or self.args.generation_batch_size != 96
            or self.args.num_train_epochs != 1 or self.args.max_steps > 0
            or self.args.dataset_shuffle or self.args.train_dataloader_shuffle
            or self.accelerator.num_processes != 2
        ):
            raise ValueError("Clinical GRPO requires the original 5,376-row ordered mixture")
        self.ref_model.eval().requires_grad_(False)
        self.add_callback(ExactEpochGuard())

    def get_train_dataloader(self, skip_batches=0):
        dataloader = super().get_train_dataloader()
        if skip_batches:
            from accelerate import skip_first_batches

            dataloader = skip_first_batches(dataloader, skip_batches)
        return dataloader

    def _postprocess_batch(self, samples, batch_encoded_inputs):
        super()._postprocess_batch(samples, batch_encoded_inputs)
        for chunk, encoded in zip(self.split_by_mini_batches(samples), batch_encoded_inputs):
            batch = encoded["grpo_batch"]
            flags = [bool(sample.extra.get("_native_opd_padding", False)) for sample in chunk]
            padding = torch.tensor(flags, dtype=torch.bool, device=batch.completion_mask.device)
            active = batch.completion_mask & ~padding[:, None]
            if batch.optimization_mask is not None:
                active &= batch.optimization_mask
            batch.optimization_mask = active


class MixedAnswerReward:
    def __init__(self, args=None, **kwargs):
        cache = Path(os.environ["MIXED_JUDGE_CACHE_DIR"])
        rank = os.environ.get("RANK", "0")
        self.judge = OpenAnswerJudge(cache / f"rank-{rank}.sqlite3")

    async def __call__(self, completions, solution, **kwargs):
        count = len(completions)

        def batch(key, default):
            values = kwargs.get(key, [default] * count)
            if len(values) != count:
                raise ValueError(f"Reward column {key} does not match the rollout batch")
            return values

        kinds = batch("reward_type", "")
        golds = batch("gold_answer", "")
        questions = batch("question", "")
        finishes = batch("finish_reason", None)
        truncated = batch("is_truncated", False)
        padding = batch("_native_opd_padding", False)
        rewards = [0.0] * count
        open_items = []
        if len(solution) != count:
            raise ValueError("Reference answers do not match the rollout batch")
        for i, text in enumerate(completions):
            if padding[i]:
                continue
            if kinds[i] == "open":
                answer, found = extract_boxed_answer(text)
                if found and finishes[i] in (None, "", "stop") and not truncated[i]:
                    open_items.append((i, questions[i], solution[i], answer))
            else:
                rewards[i] = local_score(
                    text, kinds[i], solution[i], golds[i], finishes[i], truncated[i]
                )["reward"]
        if open_items:
            semaphore = asyncio.Semaphore(self.judge.concurrency)
            timeout = aiohttp.ClientTimeout(total=self.judge.timeout)
            connector = aiohttp.TCPConnector(limit=max(32, 2 * self.judge.concurrency))
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                scores = await asyncio.gather(*(
                    self.judge.score(session, semaphore, question, reference, answer)
                    for _, question, reference, answer in open_items
                ))
            for (i, *_), score in zip(open_items, scores):
                rewards[i] = score
        return rewards


TrainerFactory.TRAINER_MAPPING["grpo"] = "mixed_grpo_plugin.MixedGRPOTrainer"
orms["mixed_xray_answer"] = MixedAnswerReward
