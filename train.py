"""Launch the DiCT paper recipes through the bundled Swift runtime."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLUGINS = ROOT / "training"
SWIFT = ROOT / "vendor" / "ms-swift"
CLINFUSION = ROOT / "vendor" / "clinfusion"
GPU_COUNTS = {"teacher-sft": 4, "sft": 4, "opd": 3, "grpo": 2, "rollout": 1}


def cli_flags(options):
    for key, value in options.items():
        yield f"--{key}"
        if isinstance(value, list):
            yield from map(str, value)
        elif isinstance(value, bool):
            yield str(value).lower()
        elif isinstance(value, dict):
            yield json.dumps(value)
        else:
            yield str(value)


def command_for(args):
    stage = args.stage
    settings = json.loads((ROOT / "configs" / f"{stage}.json").read_text())
    cuda_devices = args.gpus.split(",")
    count = GPU_COUNTS[stage]
    if len(cuda_devices) != count or len(set(cuda_devices)) != count or not all(
        device.isdecimal() for device in cuda_devices
    ):
        raise ValueError(f"{stage} requires {count} distinct GPU indices")

    env = {
        "PYTHONPATH": os.pathsep.join(str(path) for path in (SWIFT, CLINFUSION, PLUGINS)),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "NCCL_NVLS_ENABLE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_USE_MODELSCOPE": "False",
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
    }
    settings.update(model=args.model, load_args=False)
    if stage == "rollout":
        route = "rollout"
        settings.update(port=args.port, vllm_limit_mm_per_prompt={"image": 2 if args.for_stage == "opd" else 6})
        if args.for_stage == "grpo":
            settings["max_pixels"] = 262144
    else:
        route = "sft" if stage.endswith("sft") else "rlhf"
        env.update(NPROC_PER_NODE=str(count), MASTER_PORT=str(args.master_port), MASTER_ADDR="127.0.0.1",
                   NNODES="1", NODE_RANK="0")
        settings.update(dataset=args.data, output_dir=args.output)
        if stage in ("opd", "grpo"):
            settings.update(ref_model=args.model, vllm_server_port=args.port, vllm_server_group_port=args.group_port)
        if stage in ("teacher-sft", "opd"):
            env.update(CLINFUSION_MODEL_DIR=str(args.teacher if stage == "opd" else args.model),
                       CLINFUSION_BASE_DIR=str(args.clinfusion_processor),
                       CLINFUSION_VOLUME_CACHE=str(args.volume_cache))
            settings["external_plugins"] = [PLUGINS / "clinfusion_plugin.py"]
        if stage == "teacher-sft":
            settings["deepspeed"] = ROOT / "configs" / "teacher_zero3.json"
        if stage == "sft":
            env["IMAGE_MAX_TOKEN_NUM"] = "512"
        if stage == "opd":
            settings["external_plugins"].append(PLUGINS / "native_opd_plugin.py")
            settings.update(teacher_model=args.teacher, chord_sft_dataset=args.anchor)
            env.update(NATIVE_EXPECTED_REAL_ROWS="40938", NATIVE_GRPO_ONLY="0",
                       NATIVE_REWARD_GUIDED_OPD="0", NATIVE_OPDVR="0", NATIVE_XRAY_GRPO_COEF="0",
                       NATIVE_XRAY_OPD_COEF="0", NATIVE_OPD_ANSWER_WEIGHT="1", NATIVE_GRPO_SCOPE="xray")
        if stage == "grpo":
            settings["external_plugins"] = [PLUGINS / "mixed_grpo_plugin.py"]
            env.update(MIXED_JUDGE_CACHE_DIR=str(args.output / "judge_cache"),
                       VLLM_WORKER_MULTIPROC_METHOD="spawn",
                       XRAY_GRPO_JUDGE_CONCURRENCY=os.environ.get("XRAY_GRPO_JUDGE_CONCURRENCY", "32"))
    return [sys.executable, "-m", "swift.cli.main", route, *cli_flags(settings)], env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, count in GPU_COUNTS.items():
        entry = stages.add_parser(stage)
        entry.add_argument("--model", type=Path, required=True, help="Local model or checkpoint directory")
        entry.add_argument("--gpus", required=True, help=f"Comma-separated indices of {count} GPUs")
        entry.add_argument("--dry-run", action="store_true", help="Print the command without starting a process")
        if stage != "rollout":
            entry.add_argument("--data", type=Path, required=True)
            entry.add_argument("--output", type=Path, required=True)
            entry.add_argument("--master-port", type=int, default=29500)
        if stage in ("rollout", "opd", "grpo"):
            entry.add_argument("--port", type=int, default=9300, help="Local vLLM rollout port")
        if stage in ("opd", "grpo"):
            entry.add_argument("--group-port", type=int, default=52000, help="Weight synchronization port")
        if stage == "rollout":
            entry.add_argument("--for-stage", choices=("opd", "grpo"), required=True)
        if stage in ("teacher-sft", "opd"):
            entry.add_argument("--clinfusion-processor", type=Path, required=True,
                               help="Local MedSAM-Agent-Qwen3-VL-8B-MedSAM2 processor/config directory")
            entry.add_argument("--volume-cache", type=Path, required=True)
        if stage == "opd":
            entry.add_argument("--teacher", type=Path, required=True, help="Frozen ClinFusion SFT checkpoint")
            entry.add_argument("--anchor", type=Path, required=True, help="Original structured SFT JSONL")
    args = parser.parse_args()
    try:
        command, env = command_for(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(shlex.join([*(f"{key}={value}" for key, value in env.items()), *command]))
        return

    for option in ("model", "teacher", "clinfusion_processor"):
        path = getattr(args, option, None)
        if path is not None and not (path / "config.json").is_file():
            parser.error(f"--{option.replace('_', '-')} needs a local model directory with config.json")
    for option in ("data", "anchor"):
        path = getattr(args, option, None)
        if path is not None and not path.is_file():
            parser.error(f"--{option} needs an existing JSONL file")
    os.execve(sys.executable, command, os.environ | env)


if __name__ == "__main__":
    main()
