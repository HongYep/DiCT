"""Register the CT-guided OPD trainer for Swift's RLHF entry point."""

from swift.trainers.trainer_factory import TrainerFactory

TrainerFactory.TRAINER_MAPPING["grpo"] = "native_opd_trainer.NativeOPDTrainer"
