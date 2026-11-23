import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from open_clip.factory import get_model_config
from open_clip.transform import image_transform
from safetensors.torch import load_file as safe_load_file
from transformers import AutoConfig, AutoProcessor, BitImageProcessor, Dinov2Config, Dinov2Model
from transformers import PretrainedConfig, PreTrainedModel
from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model, is_deepspeed_zero3_enabled

from swift.model import (
    Model,
    ModelGroup,
    ModelLoader,
    ModelMeta,
    MultiModelKeys,
    register_model,
    register_model_arch,
)
from swift.template import Template, TemplateMeta, register_template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.templates.qwen import Qwen3VLTemplate
from swift.template.utils import findall

from custom_model.image_processing_multi_enc_qwen2_vl import ImageMultiProcessor
from custom_model.medical_image_preprocessing import sample_slices_from_volume_to_pil_fast
from custom_model.medical_vit import create_ssl_large_p8_3d_pe_model
from custom_model.modeling_multi_enc_qwen3_vl import (
    Qwen3VL_VisionWrapper,
    MultiEnc_Qwen3VLForConditionalGeneration,
    MultiEnc_Qwen3VLModel,
)
from custom_model.modeling_multi_enc_qwen2_5_vl import VisionWrapperConfig
from custom_model.openclip_encoder import CLIP, OpenCLIPVisionTower


MODEL_TYPE = "clinfusion"
TEMPLATE_TYPE = "clinfusion_3d"
MODEL_DIR = Path(os.environ["CLINFUSION_MODEL_DIR"])
BASE_DIR = Path(os.environ["CLINFUSION_BASE_DIR"])
VOLUME_CACHE = Path(os.environ["CLINFUSION_VOLUME_CACHE"])


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _model_args() -> SimpleNamespace:
    config = _read_json(MODEL_DIR / "config.json")
    processor_config = _read_json(MODEL_DIR / "preprocessor_config.json")
    vision = config["vision_config"]
    encoder_types = [name for name in vision["vision_model_config"] if name != "original_enc"]
    values = {
        "new_encoder_type": encoder_types[0],
        "new_encoder_2_type": encoder_types[1] if len(encoder_types) > 1 else None,
        "new_encoder_3_type": encoder_types[2] if len(encoder_types) > 2 else None,
        "multi_enc_fusion_type": vision["fusion_type"],
        "convnext_image_size": vision["vision_model_config"]["convnext"]["model_cfg"]["vision_cfg"]["image_size"],
        "use_fast_tokenizer": True,
        "CT_volume_cache": str(VOLUME_CACHE.parent),
    }
    values.update({
        key: processor_config[key]
        for key in [
            "merlin_patch_size",
            "merlin_image_size",
            "merlin_num_channels",
            "merlin_volume_depth",
            "merlin_sample_images",
        ]
    })
    return SimpleNamespace(**values)


def _build_convnext(config: dict) -> OpenCLIPVisionTower:
    tower = OpenCLIPVisionTower.__new__(OpenCLIPVisionTower)
    nn.Module.__init__(tower)
    tower.is_loaded = True
    tower.vision_tower_name = "convnext-large-d-320"
    tower.model_path = ""
    tower.vision_config = deepcopy(config)
    tower.is_droppath = True
    tower.model_type = "convnext_large_d_320"
    tower.model_channel = [192, 384, 768, 1536]

    clip_model = CLIP(**get_model_config(tower.model_type), drop_path=True)
    clip_model.visual.trunk.norm_pre = None
    clip_model.visual.trunk.head = None
    clip_model.visual.head = None
    tower.vision_stem = clip_model.visual.trunk.stem
    tower.vision_stages = clip_model.visual.trunk.stages
    return tower


def _build_visual_wrapper(model: MultiEnc_Qwen3VLForConditionalGeneration) -> Qwen3VL_VisionWrapper:
    saved_config = _read_json(MODEL_DIR / "config.json")["vision_config"]
    vision_configs = deepcopy(saved_config["vision_model_config"])
    original_enc = model.model.visual
    dinov2 = Dinov2Model(Dinov2Config.from_dict(vision_configs["dinov2"]))
    convnext = _build_convnext(vision_configs["convnext"])
    pe_3d, _ = create_ssl_large_p8_3d_pe_model(
        "", patch_size=_model_args().merlin_patch_size, return_config=True)
    wrapper_config = VisionWrapperConfig(
        vision_model_config=vision_configs,
        llm_hidden_size=saved_config["llm_hidden_size"],
        tokens_per_second=saved_config.get("tokens_per_second"),
        spatial_merge_size=saved_config["spatial_merge_size"],
        fusion_type=saved_config["fusion_type"],
    )
    return Qwen3VL_VisionWrapper(
        config=wrapper_config,
        original_enc=original_enc,
        dinov2=dinov2,
        convnext=convnext,
        pe_3d=pe_3d,
    )


def _load_clinfusion_checkpoint(model: PreTrainedModel) -> None:
    index = _read_json(MODEL_DIR / "model.safetensors.index.json")
    checkpoint_keys = set(index["weight_map"])
    model_keys = set(model.state_dict())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"ClinFusion checkpoint mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}")

    for shard_name in sorted(set(index["weight_map"].values())):
        state_dict = safe_load_file(MODEL_DIR / shard_name, device="cpu")
        if is_deepspeed_zero3_enabled():
            errors, _ = _load_state_dict_into_zero3_model(model, state_dict)
            if errors:
                raise RuntimeError(f"Failed loading {shard_name}: {errors[:5]}")
        else:
            result = model.load_state_dict(state_dict, strict=False)
            if result.unexpected_keys:
                raise RuntimeError(f"Unexpected keys in {shard_name}: {result.unexpected_keys[:20]}")
        del state_dict


register_model_arch(MultiModelKeys(
    MODEL_TYPE,
    language_model=["model.language_model", "lm_head"],
    vision_tower="model.visual",
    aligner=[],
    mlp="model.language_model.layers.{}.mlp",
))


class ClinFusionLoader(ModelLoader):
    def get_config(self, model_dir: str) -> PretrainedConfig:
        return AutoConfig.from_pretrained(BASE_DIR, trust_remote_code=False)

    def get_processor(self, model_dir: str, config: PretrainedConfig):
        processor = AutoProcessor.from_pretrained(BASE_DIR, trust_remote_code=False)
        args = _model_args()
        saved_config = _read_json(MODEL_DIR / "config.json")["vision_config"]["vision_model_config"]
        dinov2_processor = BitImageProcessor(
            crop_size={"height": 224, "width": 224},
            do_center_crop=True,
            do_convert_rgb=True,
            do_normalize=True,
            do_rescale=True,
            do_resize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            resample=3,
            rescale_factor=1 / 255,
            size={"shortest_edge": 256},
        )
        convnext_config = saved_config["convnext"]
        convnext_processor = image_transform(
            image_size=args.convnext_image_size,
            is_train=False,
            mean=tuple(convnext_config["preprocess_cfg"]["mean"]),
            std=tuple(convnext_config["preprocess_cfg"]["std"]),
        )
        original_image_processor = processor.image_processor
        processor.image_processor = ImageMultiProcessor(
            original_image_processor,
            new_processor_1=dinov2_processor,
            new_processor_1_type="dinov2",
            new_processor_2=convnext_processor,
            new_processor_2_type="convnext",
            new_processor_3=None,
            new_processor_3_type="pe_3d",
            merlin_patch_size=args.merlin_patch_size,
            merlin_image_size=args.merlin_image_size,
            merlin_num_channels=args.merlin_num_channels,
            merlin_volume_depth=args.merlin_volume_depth,
            merlin_sample_images=args.merlin_sample_images,
            CT_volume_cache=str(VOLUME_CACHE),
        )
        processor.image_processor.patch_size = original_image_processor.patch_size
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        return processor

    def get_model(self, model_dir: str, config: PretrainedConfig, processor, model_kwargs) -> PreTrainedModel:
        kwargs = dict(model_kwargs)
        kwargs.pop("device_map", None)
        kwargs.pop("low_cpu_mem_usage", None)
        model = MultiEnc_Qwen3VLForConditionalGeneration._from_config(config, **kwargs)
        model.set_model(MultiEnc_Qwen3VLModel.from_qwen_model(model.model))
        visual = _build_visual_wrapper(model).to(dtype=model.dtype, device=model.device)
        model.model.set_visual(visual)
        model.config.vision_config = visual.config
        _load_clinfusion_checkpoint(model)

        inference_device = os.environ.get("CLINFUSION_INFERENCE_DEVICE")
        if inference_device:
            model.to(inference_device)

        # This is an independent guard in addition to ms-swift's freeze_vit behavior.
        model.model.visual.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and not (
                    name.startswith("model.language_model.") or name.startswith("lm_head.")):
                raise RuntimeError(f"Unexpected trainable parameter outside the LLM: {name}")
        return model


register_model(ModelMeta(
    MODEL_TYPE,
    [ModelGroup([Model("Alibaba-DAMO-Academy/ClinFusion-8B")])],
    ClinFusionLoader,
    template=TEMPLATE_TYPE,
    is_multimodal=True,
    model_arch=MODEL_TYPE,
    architectures=["MultiEnc_Qwen3VLForConditionalGeneration"],
    requires=["transformers>=4.57", "open_clip_torch", "monai", "nibabel"],
    tags=["vision", "3d-medical"],
))


def _preprocess_volume(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        array = np.load(path)
        volume = torch.from_numpy(array).contiguous().float()
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        if volume.ndim != 4 or volume.shape[0] != 1:
            raise ValueError(f"Expected a (1, D, H, W) volume in {path}, got {tuple(volume.shape)}")
        if tuple(volume.shape[-3:]) != (32, 256, 256):
            volume = torch.nn.functional.interpolate(
                volume.unsqueeze(0),
                size=(32, 256, 256),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        return volume

    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        Resized,
        ScaleIntensityRanged,
        Spacingd,
        ToTensord,
    )

    VOLUME_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = VOLUME_CACHE / f"{Path(path).name.removesuffix('.nii.gz')}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    transform = Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(32, 256, 256), mode="trilinear"),
        ToTensord(keys=["image"]),
    ])
    volume = transform({"image": path})["image"].as_tensor().contiguous().float()
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(volume, tmp_path)
    os.replace(tmp_path, cache_path)
    return volume


class ClinFusion3DTemplate(Qwen3VLTemplate):
    load_images = True

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        volume_paths = []
        volume_suffixes = (".nii", ".nii.gz", ".npy")
        if isinstance(inputs.channel, str) and inputs.channel.endswith(volume_suffixes):
            volume_paths.append(inputs.channel)
        volume_paths.extend([
            image for image in inputs.images
            if isinstance(image, str) and image.endswith(volume_suffixes)
        ])
        if volume_paths:
            nifti_images = [
                image for image in inputs.images
                if isinstance(image, str) and image.endswith(volume_suffixes)
            ]
            if nifti_images and len(nifti_images) != len(inputs.images):
                raise ValueError("ClinFusion 3D batches cannot mix NIfTI and ordinary images in one sample")
            volumes = [_preprocess_volume(path) for path in volume_paths]
            slices: List[Image.Image] = []
            for volume in volumes:
                slices.extend(sample_slices_from_volume_to_pil_fast(
                    volume, num_slices=4, random=False))
            inputs.images = slices
            inputs.mm_processor_kwargs["clinfusion_volumes"] = volumes
        super()._preprocess_inputs(inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        # `channel` is only a carrier for the NIfTI path. It has already been
        # converted to visual tensors in _preprocess_inputs and must not reach
        # transformers.generate().
        encoded.pop("channel", None)
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        loss_scale = encoded.get("loss_scale")
        mm_mask = [False] * len(input_ids)

        if inputs.images:
            volumes = inputs.mm_processor_kwargs.get("clinfusion_volumes")
            kwargs = {"volume": volumes} if volumes else {}
            media_inputs = self.processor.image_processor(
                images=inputs.images,
                return_tensors="pt",
                do_resize=False,
                **kwargs,
            )
            grid_thw = media_inputs["image_grid_thw"]
            idx_list = findall(input_ids, self.image_token_id)
            merge_length = self.processor.image_processor.merge_size**2

            def new_tokens(i: int) -> List[int]:
                token_len = grid_thw[i].prod() // merge_length
                return [self.image_token_id] * int(token_len)

            input_ids, labels, loss_scale, mm_mask = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, new_tokens, mm_mask=mm_mask)
            encoded.update(media_inputs)

        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        encoded["loss_scale"] = loss_scale
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = super()._data_collator_mm_data(batch)
        for key in ["pixel_values_dinov2", "pixel_values_convnext", "pixel_values_pe_3d"]:
            values = [item[key] for item in batch if item.get(key) is not None]
            if values:
                result[key] = torch.cat(values, dim=0)
        return result

    def _data_collator(
        self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None
    ) -> Dict[str, Any]:
        result = super()._data_collator(batch, padding_to=padding_to)
        if not self.is_training:
            # Transformers 5.3 requires this field while computing multimodal
            # RoPE positions, but the older custom ClinFusion forward signature
            # cannot accept it during generate().
            result.pop("mm_token_type_ids", None)
            result.pop("channel", None)
        return result


register_template(TemplateMeta(
    TEMPLATE_TYPE,
    prefix=[],
    prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
    chat_sep=["<|im_end|>\n"],
    suffix=["<|im_end|>"],
    system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
    default_system=None,
    stop_words=["<|endoftext|>"],
    template_cls=ClinFusion3DTemplate,
))
