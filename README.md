# DiCT training code

DiCT transfers CT-informed reasoning to X-ray vision-language models. The training recipe supports Qwen3-VL-4B-Instruct and HuluV2-4B. At inference, the student takes X-rays and a question.

| Stage | Input | Training setting |
| --- | --- | --- |
| ClinFusion teacher SFT | 30,662 CT examples with reasoning targets | Four GPUs, batch size 64, one epoch at 1e-5 |
| Structured student SFT | 10,113 paired-view X-ray examples | Four GPUs, batch size 64, one epoch at 1e-5 |
| CT-guided OPD | 40,938 paired X-ray/CT questions and 54 zero-loss padding rows | Three training GPUs and one vLLM rollout GPU, 96 questions per step, 427 steps at 1e-6 |
| Clinical GRPO | 4,027 ReXVQA Measurement questions, 1,344 3D-RAD questions and 5 zero-loss padding rows | Two training GPUs and one vLLM rollout GPU, 12 questions per step, 8 rollouts per question, 448 steps at 1e-6 |

The recipes use bf16 and DeepSpeed ZeRO-3, with the vision encoder and aligner frozen. During OPD, the frozen ClinFusion teacher scores aligned CT reasoning and answer tokens at weight 0.5. The structured SFT checkpoint supplies the frozen X-ray reference at weight 0.02. The original SFT targets supply a format-only replay loss at weight 0.1. During clinical GRPO, the OPD checkpoint initializes the policy and supplies the frozen KL reference at weight 0.02.

## Environment

Use Linux, Python 3.12 and CUDA GPUs. The paper used NVIDIA H20 GPUs. Install CUDA-compatible PyTorch and vLLM builds together, then install the Python dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.13.0' 'torchvision==0.28.0' 'vllm==0.27.1'
python -m pip install -r requirements.txt
python -m pip check
```

Use the wheel index appropriate for your CUDA build if needed. The training process also needs the vLLM client libraries for weight synchronization. `train.py` selects the bundled ms-swift runtime with the trainer interfaces used by DiCT.

## Training inputs

Supply local model directories, media and stage-ready JSONL files from your authorized copies of the assets. Start from the base student and the ClinFusion-8B SFT checkpoint, or train ClinFusion-8B with the teacher recipe below. ClinFusion also needs the MedSAM-Agent-Qwen3-VL-8B-MedSAM2 config and processor files.

The files in [examples](examples) contain fabricated rows for checking each schema. Use `prepare_data.py` to validate a manifest and write a new one with resolved media paths. It preserves row order and the provided targets.

```bash
python prepare_data.py sft data/student-structured.jsonl --structured --paper \
  --media-root data --output data/student-structured.local.jsonl
python prepare_data.py opd examples/opd.jsonl --skip-media-check
```

| Manifest | Required content |
| --- | --- |
| Teacher SFT | `messages` with CT question and reasoning/answer target. `channel` holds the CT volume path. |
| Student SFT and OPD replay | `messages` with X-ray question and structured target. `images` lists the two radiographs in prompt order. The target uses `[thinking with XRay]`, `[thinking with CT volume]`, and `[Final Answer]` with one `\boxed{...}` answer. |
| OPD | X-ray user message and two `images`, paired CT question in `teacher_prompt`, and the volume in `teacher_images`. Match each X-ray question to the corresponding CT case. Set `_native_opd_padding` on the 54 batch-fill rows. |
| GRPO | X-ray user message and `images`, answer in `solution`, `source_dataset`, and `reward_type`. Use `gold_answer` for choice rows and `question` for open-answer rows. Preserve the prepared 3:1 source order and the five `_native_opd_padding` rows. |

Radiograph user messages use one `<image>` per file. GRPO reward types are `rex_scalar`, `3d_measurement`, `choice`, and `open`. ReXVQA contributes its open-ended Measurement training subset. For `rex_scalar`, write the reference length in millimeters as a numeric string in `solution` (for example, `"20.0"`). For `3d_measurement`, include the unit in the reference and expected answer. The 3D-RAD part contains 538 open questions, 269 measurements, and 537 four-choice questions. The paper's exact results require its prepared splits, synthesized reasoning targets, question prompts, media and matching checkpoints.

## Launch training

Run the commands from the repository root. Set the paths to your local assets. Repeat the student commands with the other backbone and a different output name. Checkpoint paths below use `<run>` to stand for the version directory printed by Swift.

```bash
STUDENT_BASE=models/Qwen3-VL-4B-Instruct   # Or the HuluV2-4B model directory
PROCESSOR=models/MedSAM-Agent-Qwen3-VL-8B-MedSAM2
VOLUME_CACHE=output/diagnostics/ct-volume-cache
STRUCTURED_DATA=data/student-structured.local.jsonl
OPD_DATA=data/opd.local.jsonl
GRPO_DATA=data/grpo.local.jsonl
```

### Teacher SFT and student SFT

Skip teacher SFT when a matching ClinFusion-8B-SFT checkpoint is available.

```bash
python train.py teacher-sft --gpus 0,1,2,3 \
  --model models/ClinFusion-8B --clinfusion-processor "$PROCESSOR" \
  --volume-cache "$VOLUME_CACHE" --data data/teacher.local.jsonl \
  --output output/ct/teacher-sft

python train.py sft --gpus 0,1,2,3 --model "$STUDENT_BASE" \
  --data "$STRUCTURED_DATA" --output output/xray/student-sft
```

Pass the X-ray-only reasoning manifest to `train.py sft` to run the direct SFT control.

### CT-guided OPD

Set `SFT_CHECKPOINT` in both terminals and `TEACHER_CHECKPOINT` in the trainer terminal.

```bash
SFT_CHECKPOINT='output/xray/student-sft/<run>/checkpoint-158'
TEACHER_CHECKPOINT='output/ct/teacher-sft/<run>/checkpoint-480'
```

Start the rollout server in a separate terminal.

```bash
python train.py rollout --for-stage opd --gpus 0 --model "$SFT_CHECKPOINT" --port 9300
```

Once the rollout server is ready at `http://127.0.0.1:9300/health/`, run:

```bash
python train.py opd --gpus 1,2,3 --model "$SFT_CHECKPOINT" \
  --teacher "$TEACHER_CHECKPOINT" --clinfusion-processor "$PROCESSOR" \
  --volume-cache "$VOLUME_CACHE" --data "$OPD_DATA" --anchor "$STRUCTURED_DATA" \
  --port 9300 --output output/xray/student-opd
```

Stop the OPD rollout server after training finishes.

### Clinical GRPO

Set `OPD_CHECKPOINT` to the completed OPD checkpoint in both terminals.

```bash
OPD_CHECKPOINT='output/xray/student-opd/<run>/checkpoint-427'
```

Start a GRPO rollout server on one GPU.

```bash
python train.py rollout --for-stage grpo --gpus 0 --model "$OPD_CHECKPOINT" --port 9301
```

Serve the frozen Qwen3.8-27B judge on a separate device or host with an OpenAI-compatible Chat Completions endpoint. The server must accept JSON schema responses and the Qwen thinking control in the request. The reward code sends the question, reference answer and extracted candidate answer to this endpoint.

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve models/Qwen3.8-27B \
  --host 127.0.0.1 --port 8400 --served-model-name Qwen3.8-27B --dtype bfloat16
```

```bash
export XRAY_GRPO_JUDGE_ENDPOINT=http://127.0.0.1:8400/v1/chat/completions
export XRAY_GRPO_JUDGE_MODEL=Qwen3.8-27B

python train.py grpo --gpus 1,2 --model "$OPD_CHECKPOINT" \
  --data "$GRPO_DATA" --port 9301 --output output/xray/student-grpo
```

For an authenticated judge, set `XRAY_GRPO_JUDGE_API_KEY` in the trainer's environment. After training, stop the rollout and judge servers. Append `--dry-run` to any `train.py` invocation to print the Swift command without loading models or starting a job.

## Source and licenses

The release contains the training entry points, student and teacher plugins, reward functions, and the runtime source needed to run the recipes. See [third-party notices](THIRD_PARTY.md) for bundled components and licenses.
