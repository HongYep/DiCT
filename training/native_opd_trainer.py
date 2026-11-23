"""Native ClinFusion CT+answer scoring with SFT format replay.

Reuse Swift's sampling, separate teacher template, reference forward, ZeRO-3,
and CHORD dataset loader. Override only routing, native scoring, and the loss.
The CHORD *mixture loss* is not used: this is an additive masked SFT anchor.
"""

from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import os

import torch
from accelerate.utils import gather_object

from swift.rl_core.data import GRPOSample
from swift.rlhf_trainers import GRPOTrainer
from swift.rlhf_trainers.gkd_helpers import encode_teacher_view
from swift.rlhf_trainers.utils import (build_routed_token_mask, collate_to_grpo_micro_batch,
                                      get_even_process_data, encode_sample)
from swift.trainers import disable_gradient_checkpointing
from swift.utils import to_device, disable_deepspeed_zero3

from native_cot import build_native_plan, build_xray_teacher_plan, sft_anchor_mask
from native_opd_loss import (native_opd_losses, opdvr_gate_delta, xray_grpo_loss,
                             xray_reference_kl_loss)
from structured_cot import flatten_token_ids, restore_token_shape


class NativeOPDTrainer(GRPOTrainer):
    xray_grpo_coefficient = 0.0
    grpo_scope = 'xray'
    xray_opd_coefficient = 0.0
    grpo_only = False
    teacher_answer_weight = 1.0
    reward_guided_opd = False
    opdvr_enabled = False

    def _setup_teacher(self):
        super()._setup_teacher()
        if self.grpo_only:
            if self._has_teacher_explicit():
                raise ValueError('GRPO ablation must not configure an explicit teacher')
            # Old paired datasets carry teacher_prompt. Disable Swift's implicit
            # student-as-teacher route as well as the explicit CT teacher.
            self._is_dynamic_self_distillation = False
            self._is_self_distillation = False
            self._has_teacher = False

    def _prepare_rollout_params(self):
        super()._prepare_rollout_params()
        # Swift sends request_config to the generation backend, but uses
        # self.temperature to divide logits in student/reference/teacher scoring.
        # Greedy decoding is argmax; its temperature=0 must never be a divisor.
        self.generation_temperature = self.request_config.temperature
        self.logprob_temperature = 1.0 if self.generation_temperature == 0 else self.generation_temperature
        self.temperature = self.logprob_temperature

    def __init__(self, *args, **kwargs):
        self.grpo_only = os.environ.get('NATIVE_GRPO_ONLY', '0') == '1'
        self.reward_guided_opd = os.environ.get('NATIVE_REWARD_GUIDED_OPD', '0') == '1'
        self.opdvr_enabled = os.environ.get('NATIVE_OPDVR', '0') == '1'
        self.teacher_answer_weight = float(os.environ.get('NATIVE_OPD_ANSWER_WEIGHT', '1'))
        if not 0 < self.teacher_answer_weight < float('inf'):
            raise ValueError('NATIVE_OPD_ANSWER_WEIGHT must be finite and positive')
        super().__init__(*args, **kwargs)
        # These are the existing experiment's supported settings. Reject routes
        # that re-collate/slice away the plugin's semantic alignment metadata.
        if (self.dynamic_num_samples or self.use_teacher_api
                or self.template.sequence_parallel_size != 1 or self.template.padding_free
                or self.use_liger_loss or self.args.async_generate):
            raise ValueError("Native OPD requires synchronous local teacher, padded batches, SP=1, fixed-size rollouts")
        self.xray_grpo_coefficient = float(os.environ.get('NATIVE_XRAY_GRPO_COEF', '0'))
        self.grpo_scope = os.environ.get('NATIVE_GRPO_SCOPE', 'xray')
        if self.grpo_scope not in ('xray', 'full'):
            raise ValueError('NATIVE_GRPO_SCOPE must be xray or full')
        if not 0 <= self.xray_grpo_coefficient < float('inf'):
            raise ValueError('NATIVE_XRAY_GRPO_COEF must be finite and nonnegative')
        if self.loss_type != 'grpo' or self.num_iterations != 1:
            raise ValueError("Native OPD requires loss_type=grpo and num_iterations=1")
        if self.xray_grpo_coefficient:
            if not self.reward_funcs or self.num_generations < 2 or self.generation_temperature <= 0:
                raise ValueError('XRay GRPO requires answer rewards, multiple rollouts, and positive temperature')
            if (self.advantage_estimator != 'grpo' or self.scale_rewards != 'group'
                    or self.kl_in_reward or self.importance_sampling_level != 'token'):
                raise ValueError('XRay GRPO requires grouped advantages, token ratios, and separate reference KL')
        elif self.reward_funcs and not self.opdvr_enabled:
            raise ValueError('Pure native OPD must not receive reward functions unless OPDVR is enabled')
        if self.opdvr_enabled:
            if (self.grpo_only or self.xray_grpo_coefficient or self.num_generations != 1
                    or self.dynamic_sample or self.reward_guided_opd):
                raise ValueError('OPDVR requires teacher OPD only, one rollout, no resampling, and no reward-guided mask')
            if (len(self.reward_funcs) != 1
                    or self.reward_funcs[0].__class__.__name__ != 'OPDVRClosedAnswerReward'):
                raise ValueError('OPDVR requires exactly the native_opdvr_closed binary verifier')
            print('OPDVR enabled: closed answers use binary correctness-gated teacher OPD; '
                  'open answers retain vanilla OPD.', flush=True)
        if self.dynamic_sample and (not self.xray_grpo_coefficient or self.max_resample_times < 1):
            raise ValueError('Native dynamic sampling requires answer GRPO and max_resample_times >= 1')
        if self.reward_guided_opd:
            if self.grpo_only or not self.xray_grpo_coefficient or not self.reward_funcs:
                raise ValueError('Reward-guided OPD requires joint OPD+GRPO with answer rewards')
            print('Reward-guided OPD enabled: exact-correct closed responses skip CT teacher OPD; open responses retain it.', flush=True)
        if self.grpo_only:
            if (self.teacher_model is not None or self.teacher_template is not None
                    or self.teacher_kl_coef != 0 or not self.xray_grpo_coefficient
                    or float(os.environ.get('NATIVE_XRAY_OPD_COEF', '0')) != 0):
                raise ValueError('GRPO ablation requires no teacher, zero OPD coefficients, and answer GRPO')
            print('GRPO-only ablation: CT/XRay OPD disabled; no distillation teacher loaded.', flush=True)
        elif self.teacher_template is None or self.teacher_model is None:
            raise ValueError("Native OPD requires a local teacher template and frozen original SFT reference")
        if self.ref_model is None:
            raise ValueError('A frozen original SFT reference is required')
        if self.chord_sft_iterator is None:
            raise ValueError("Provide the original student SFT data with --chord_sft_dataset")
        self.sft_anchor_coefficient = self.args.chord_mu_peak
        if self.sft_anchor_coefficient is None or self.sft_anchor_coefficient <= 0:
            raise ValueError("Set --chord_mu_peak 0.1 for the constant additive SFT anchor")
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model.requires_grad_(False)
        expected_rows = os.environ.get('NATIVE_EXPECTED_REAL_ROWS')
        if expected_rows is not None:
            real = sum(not bool(row.get('_native_opd_padding', False)) for row in self.train_dataset)
            prompt_batch = self.args.generation_batch_size // self.num_generations
            if self.args.generation_batch_size % self.num_generations or prompt_batch < 1:
                raise ValueError('Generation batch must contain complete prompt groups')
            if real != int(expected_rows) or len(self.train_dataset) % prompt_batch:
                raise ValueError('Single-epoch dataset lost rows or has an incomplete prompt batch')
            if self.dynamic_sample or self.args.max_steps > 0 or self.args.num_train_epochs != 1:
                raise ValueError('Exact single-epoch coverage requires no resampling, one epoch and max_steps=-1')
            print(f'Single-epoch coverage: real={real}; padding={len(self.train_dataset)-real}; '
                  f'prompt_batch_size={prompt_batch}; generations={self.num_generations}; '
                  f'expected_updates={len(self.train_dataset)//prompt_batch}', flush=True)
        self._init_xray_teacher()
        if self.opdvr_enabled and self.xray_opd_coefficient:
            raise ValueError('OPDVR comparison does not support an auxiliary XRay teacher')

    def _init_xray_teacher(self):
        self.xray_opd_coefficient = float(os.environ.get('NATIVE_XRAY_OPD_COEF', '0'))
        if not 0 <= self.xray_opd_coefficient < float('inf'):
            raise ValueError('NATIVE_XRAY_OPD_COEF must be finite and nonnegative')
        if not self.xray_opd_coefficient:
            return
        from swift.model import get_model_processor
        from swift.template import get_template
        path = os.environ['NATIVE_XRAY_TEACHER_MODEL']
        devices = [int(d) for d in os.environ['NATIVE_XRAY_TEACHER_DEVICES'].split(',')]
        world = self.accelerator.num_processes
        if len(devices) != world or len(set(devices)) != world:
            raise ValueError('Provide one dedicated XRay teacher CUDA device per training rank')
        if any(d < world or d >= torch.cuda.device_count() for d in devices):
            raise ValueError('XRay teacher devices must be visible and disjoint from training ranks')
        self.xray_teacher_device = torch.device('cuda', devices[self.accelerator.process_index])
        self.xray_teacher_batch_size = int(os.environ.get('NATIVE_XRAY_TEACHER_BATCH_SIZE', '2'))
        if self.xray_teacher_batch_size < 1:
            raise ValueError('XRay teacher scoring batch size must be positive')
        # The frozen auxiliary teacher has a dedicated device, outside ZeRO and
        # the optimizer. Preserve RNG states so loading it cannot reorder data.
        with disable_deepspeed_zero3(), torch.random.fork_rng(devices=[self.accelerator.device.index,
                                                                      self.xray_teacher_device.index]):
            self.xray_teacher_model, processor = get_model_processor(
                path, model_type='qwen3_vl_moe', torch_dtype=torch.bfloat16,
                device_map={'': str(self.xray_teacher_device)}, attn_impl='sdpa')
        self.xray_teacher_model.eval().requires_grad_(False)
        self.xray_teacher_template = get_template(processor, enable_thinking=False,
                                                  max_length=8192, truncation_strategy='raise')
        self.xray_teacher_template.set_mode('train')
        self.xray_teacher_template.model = self.xray_teacher_model
        print(f'XRay OPD teacher loaded: {path}; device={self.xray_teacher_device}; '
              f'coefficient={self.xray_opd_coefficient}; frozen=True', flush=True)

    def _score_xray_teacher(self, samples, batch):
        """Score the same rollout on XRay+question, then map to student positions."""
        mapped = torch.zeros_like(batch.completion_mask, dtype=torch.float32)
        mask = torch.zeros_like(batch.completion_mask)
        groups = [[] for _ in samples]
        pending = []
        for row, sample in enumerate(samples):
            ids, _ = flatten_token_ids(sample.response_token_ids)
            plan = build_xray_teacher_plan(ids, self.processing_class,
                                           self.xray_teacher_template.tokenizer)
            positions = batch.completion_mask[row].nonzero(as_tuple=True)[0].tolist()
            if len(positions) not in (len(ids), len(ids) + 1):
                raise ValueError('XRay teacher/student completion frames differ')
            if not plan['groups']:
                continue
            teacher = replace(sample, response_token_ids=[plan['teacher_ids']],
                              response_loss_mask=[], teacher_token_mask=[], grpo_token_mask=[],
                              answer_token_mask=[], encoded=None)
            teacher.encoded = encode_sample(teacher, self.xray_teacher_template)
            teacher.encoded.pop('_extra_kwargs', None)
            labels = [int(t) for t in teacher.encoded['labels'] if int(t) != -100]
            if labels != plan['teacher_ids']:
                raise ValueError('XRay teacher template changed target labels (prefix/EOS/offset)')
            pending.append((row, teacher, plan, positions))
        for start in range(0, len(pending), self.xray_teacher_batch_size):
            chunk = pending[start:start + self.xray_teacher_batch_size]
            inputs, tb = collate_to_grpo_micro_batch(
                [item[1] for item in chunk], self.xray_teacher_template,
                device=self.xray_teacher_device, use_logits_to_keep=True,
                include_routed_masks=False)
            inputs.pop('labels', None)
            device_context = (torch.cuda.device(self.xray_teacher_device)
                              if self.xray_teacher_device.type == 'cuda' else nullcontext())
            with torch.no_grad(), device_context:
                logps = self._get_per_token_logps_and_entropies(self.xray_teacher_model, inputs, tb)[0]
            if not torch.isfinite(logps[tb.completion_mask]).all():
                raise ValueError('Non-finite XRay teacher log probabilities')
            logps = logps.detach().to(device=mapped.device, dtype=mapped.dtype)
            sources, targets, weights = [], [], []
            for j, (row, _, plan, positions) in enumerate(chunk):
                ti = tb.completion_mask[j].nonzero(as_tuple=True)[0].tolist()
                for group in plan['groups']:
                    target = [positions[i] for i in group['student']]
                    source = [ti[i] for i in group['teacher']]
                    for destination in target:
                        sources.extend(j * logps.shape[1] + i for i in source)
                        targets.extend([row * mapped.shape[1] + destination] * len(source))
                        weights.extend([1.0 / len(target)] * len(source))
                    groups[row].append(target)
            source_index = torch.tensor(sources, dtype=torch.long, device=mapped.device)
            target_index = torch.tensor(targets, dtype=torch.long, device=mapped.device)
            weight = torch.tensor(weights, dtype=logps.dtype, device=mapped.device)
            mapped.flatten().index_add_(0, target_index, logps.flatten()[source_index] * weight)
            mask.flatten()[target_index] = True
        if (mask & ~batch.grpo_token_mask).any() or (mask & batch.teacher_token_mask).any():
            raise ValueError('XRay OPD leaked outside the XRay body')
        batch.xray_teacher_per_token_logps = mapped
        batch.xray_teacher_token_mask = mask
        batch.xray_teacher_groups = groups

    def _score_completions(self, samples):
        # Swift may replace whole reward groups with newly generated samples.
        # Build semantic alignment only for the final selection, before encoding;
        # resampled responses must get their own native teacher targets and masks.
        samples = super()._score_completions(samples)
        for sample in samples:
            ids, lengths = flatten_token_ids(sample.response_token_ids)
            tokenizer = self.processing_class if self.grpo_only else self.teacher_template.tokenizer
            plan = build_native_plan(ids, self.processing_class, tokenizer)
            sample.extra['_native_opd_plan'] = plan
            teacher_mask = [0] * len(ids) if self.grpo_only else plan['teacher_opd_mask']
            sample.teacher_token_mask = restore_token_shape(teacher_mask, lengths)
            # This routed field always remains the XRay reference KL mask.
            # Full-response GRPO selects completion_mask only at loss time.
            sample.grpo_token_mask = restore_token_shape(plan['xray_ref_kl_mask'], lengths)
            sample.answer_token_mask = []
        return samples

    def _dynamic_sampling(self, samples, rewards_per_func):
        before = self.compute_std(samples, rewards_per_func)
        selected, rewards = super()._dynamic_sampling(samples, rewards_per_func)
        after = self.compute_std(selected, rewards)
        metrics = self._metrics['train']
        metrics['dynamic_sampling/zero_std_before'].append((before == 0).float().mean().item())
        metrics['dynamic_sampling/zero_std_after'].append((after == 0).float().mean().item())
        # Swift returns the original batch if it cannot fill the requested batch.
        metrics['dynamic_sampling/fallback'].append(float(selected is samples))
        return selected, rewards

    def _collate_teacher_opsd_batch(self, batch, template):
        teacher_samples = []
        plans = []
        for sample in batch:
            plan = sample.extra['_native_opd_plan']
            ids = plan['teacher_ids']
            if not ids:
                # Keep all ZeRO ranks on the same forward/collective schedule.
                # This placeholder's probabilities are NEVER used for OPD.
                ids = self.teacher_template.tokenizer.encode('No answer.', add_special_tokens=False)
            teacher = replace(sample, teacher_response_token_ids=[ids],
                              response_loss_mask=[], teacher_token_mask=[], grpo_token_mask=[])
            teacher.encoded = encode_teacher_view(teacher, self.teacher_template)
            labeled_ids = [int(t) for t in teacher.encoded['labels'] if int(t) != -100]
            if labeled_ids != ids:
                raise ValueError("Teacher template changed native target labels (prefix/EOS/off-by-one)")
            teacher_samples.append(teacher)
            plans.append(plan)
        inputs, teacher_batch = collate_to_grpo_micro_batch(
            teacher_samples, self.teacher_template, device=self.accelerator.device,
            use_logits_to_keep=True, include_routed_masks=False)
        inputs.pop('labels', None)
        teacher_batch.native_plans = plans
        return inputs, teacher_batch

    def _compute_teacher_logps(self, model_inputs, grpo_batch, origin_data=None,
                               teacher_model_inputs=None, teacher_grpo_batch=None):
        if teacher_model_inputs is None or teacher_grpo_batch is None:
            raise ValueError("Native OPD must encode teacher CT+question separately")
        with torch.no_grad(), self.load_teacher_model_context(), disable_gradient_checkpointing(
                self.teacher_model, self.args.gradient_checkpointing_kwargs):
            logps = self._get_per_token_logps_and_entropies(
                self.teacher_model, teacher_model_inputs, teacher_grpo_batch)[0]
        mapped = torch.zeros_like(grpo_batch.completion_mask, dtype=logps.dtype)
        groups = []
        sources, targets, weights = [], [], []
        for row, plan in enumerate(teacher_grpo_batch.native_plans):
            si = grpo_batch.completion_mask[row].nonzero(as_tuple=True)[0].tolist()
            ti = teacher_grpo_batch.completion_mask[row].nonzero(as_tuple=True)[0].tolist()
            student_count = len(plan['teacher_opd_mask'])
            if len(si) not in (student_count, student_count + 1):
                raise ValueError("Student completion frame differs from original sampled IDs")
            row_groups = []
            for group in plan['groups']:
                target = [si[i] for i in group['student']]
                source = [ti[i] for i in group['teacher']]
                for destination in target:
                    sources.extend(row * logps.shape[1] + i for i in source)
                    targets.extend([row * mapped.shape[1] + destination] * len(source))
                    weights.extend([1.0 / len(target)] * len(source))
                row_groups.append(target)
            groups.append(row_groups)
        source_index = torch.tensor(sources, dtype=torch.long, device=logps.device)
        target_index = torch.tensor(targets, dtype=torch.long, device=logps.device)
        weights = torch.tensor(weights, dtype=logps.dtype, device=logps.device)
        mapped.flatten().index_add_(0, target_index, logps.flatten()[source_index] * weights)
        grpo_batch.native_student_groups = groups
        return mapped

    @staticmethod
    def _reward_guided_teacher_suppression(sample, reward_row) -> bool:
        # Use the raw answer reward before group normalization. Only exact-correct
        # closed responses are exempt; open answers and partial multiselect scores
        # keep teacher OPD. Padding rows are never active training examples.
        if bool(sample.extra.get('_native_opd_padding', False)):
            return False
        if str(sample.extra.get('question_type', '')).lower() != 'closed':
            return False
        if reward_row is None or getattr(reward_row, 'numel', lambda: 0)() == 0:
            return False
        return float(reward_row[0].item()) >= 1.0 - 1e-6

    def _postprocess_batch(self, samples, batch_encoded_inputs):
        device = self.accelerator.device
        reward_guided_flags = [False] * len(samples)
        if self.reward_guided_opd:
            local_rewards = get_even_process_data(self, self._rewards_per_func)
            if len(local_rewards) != len(samples):
                raise ValueError('Reward-guided OPD rewards do not match local rollout samples')
            reward_guided_flags = [self._reward_guided_teacher_suppression(sample, reward)
                                   for sample, reward in zip(samples, local_rewards)]
            for sample, flag in zip(samples, reward_guided_flags):
                sample.extra['_native_reward_guided_skip_teacher_opd'] = bool(flag)
        opdvr_closed_flags = [False] * len(samples)
        opdvr_correct_flags = [False] * len(samples)
        if self.opdvr_enabled:
            local_rewards = get_even_process_data(self, self._rewards_per_func)
            if len(local_rewards) != len(samples):
                raise ValueError('OPDVR verifier rewards do not match local rollout samples')
            for index, (sample, reward_row) in enumerate(zip(samples, local_rewards)):
                is_closed = str(sample.extra.get('question_type', '')).lower() == 'closed'
                is_padding = bool(sample.extra.get('_native_opd_padding', False))
                if not is_closed or is_padding:
                    continue
                if reward_row is None or getattr(reward_row, 'numel', lambda: 0)() != 1:
                    raise ValueError('OPDVR expects one scalar binary verifier reward per closed response')
                reward = float(reward_row[0].item())
                if not 0.0 <= reward <= 1.0:
                    raise ValueError(f'OPDVR verifier reward is outside [0, 1]: {reward}')
                opdvr_closed_flags[index] = True
                opdvr_correct_flags[index] = reward >= 1.0 - 1e-6
                sample.extra['_native_opdvr_closed'] = True
                sample.extra['_native_opdvr_correct'] = opdvr_correct_flags[index]
        if self.xray_grpo_coefficient:
            # Reuse Swift's distributed grouping and reward normalization, but
            # do not call its postprocessor: that also adds teacher advantages.
            advantages = self._compute_advantages(samples, self._rewards_per_func, batch_encoded_inputs)
            local_advantages = get_even_process_data(self, advantages)
            if len(local_advantages) != len(samples):
                raise ValueError('Distributed reward advantages do not match local rollout samples')
            self._logs['advantages'].extend(advantages.tolist())
            for sample, advantage in zip(samples, local_advantages):
                sample.advantages = advantage.detach().to(device)
        else:
            self._logs['advantages'].extend(gather_object([0.0] * len(samples)))
            for sample in samples:
                sample.advantages = torch.zeros((), device=device)
        chunks = self.split_by_mini_batches(samples)
        if len(chunks) != len(batch_encoded_inputs):
            raise ValueError("Native OPD sample chunks and encoded batches differ")
        for samples_chunk, encoded in zip(chunks, batch_encoded_inputs):
            batch = encoded['grpo_batch']
            batch.advantages = torch.stack([s.advantages for s in samples_chunk])[:, None].expand_as(
                batch.completion_mask)
            if self.opdvr_enabled:
                batch.native_opdvr_closed_mask = torch.tensor(
                    [bool(s.extra.get('_native_opdvr_closed', False)) for s in samples_chunk],
                    dtype=torch.bool, device=device)
                batch.native_opdvr_correctness = torch.tensor(
                    [bool(s.extra.get('_native_opdvr_correct', False)) for s in samples_chunk],
                    dtype=torch.bool, device=device)
                if (batch.native_opdvr_correctness & ~batch.native_opdvr_closed_mask).any():
                    raise ValueError('OPDVR marked an open/padding response correct')
            formatting = build_routed_token_mask(
                [s.extra['_native_opd_plan']['format_token_mask'] for s in samples_chunk],
                batch.completion_mask, device)
            if self.reward_guided_opd:
                teacher_masks = []
                for sample in samples_chunk:
                    plan = sample.extra['_native_opd_plan']
                    skip = bool(sample.extra.get('_native_reward_guided_skip_teacher_opd', False))
                    teacher_masks.append([0] * len(plan['teacher_opd_mask']) if skip
                                         else plan['teacher_opd_mask'])
                batch.teacher_token_mask = build_routed_token_mask(
                    teacher_masks, batch.completion_mask, device)
            # The shared collator represents an all-empty rollout mask as None.
            # An empty response still receives the independent SFT replay loss.
            for name in ('teacher_token_mask', 'grpo_token_mask'):
                if getattr(batch, name) is None:
                    setattr(batch, name, torch.zeros_like(batch.completion_mask))
            if formatting is None:
                formatting = torch.zeros_like(batch.completion_mask)
            masks = [batch.teacher_token_mask, batch.grpo_token_mask, formatting]
            for i, left in enumerate(masks):
                for right in masks[i + 1:]:
                    if (left & right).any():
                        raise ValueError("Padded OPD/XRay-reference/format masks overlap")
            batch.native_format_token_mask = formatting
            answer_content = build_routed_token_mask(
                [s.extra['_native_opd_plan']['answer_content_mask'] for s in samples_chunk],
                batch.completion_mask, device)
            if answer_content is None:
                answer_content = torch.zeros_like(batch.completion_mask)
            if not self.grpo_only:
                suppressed = torch.tensor(
                    [bool(s.extra.get('_native_reward_guided_skip_teacher_opd', False))
                     for s in samples_chunk], device=device)[:, None]
                if (answer_content & ~batch.teacher_token_mask & ~suppressed).any():
                    raise ValueError('Boxed answer weight leaked outside aligned teacher content')
            batch.native_answer_content_mask = answer_content
            if self.xray_opd_coefficient:
                self._score_xray_teacher(samples_chunk, batch)
            # Synthetic batch-fill rows keep distributed forward schedules equal,
            # but must never train on a repeated question. Keep the forward mask
            # intact and use a separate mask for every loss term.
            padding = torch.tensor([bool(s.extra.get('_native_opd_padding', False))
                                    for s in samples_chunk], device=device)
            batch.native_loss_completion_mask = batch.completion_mask & ~padding[:, None]
            batch.native_active_fraction = (~padding).float().mean()
            if self.reward_guided_opd:
                suppressed_rows = torch.tensor(
                    [bool(s.extra.get('_native_reward_guided_skip_teacher_opd', False))
                     for s in samples_chunk], device=device)
                batch.native_reward_guided_suppressed_rows = suppressed_rows
                mode = 'train' if self.model.training else 'eval'
                self._metrics[mode]['reward_guided_suppressed_rows'].append(
                    suppressed_rows.float().sum().item())
                self._metrics[mode]['reward_guided_suppressed_tokens'].append(
                    (batch.completion_mask & suppressed_rows[:, None]).sum().item())
        stats = torch.zeros(8, dtype=torch.float64, device=device)
        for sample in samples:
            if sample.extra.get('_native_opd_padding', False):
                continue
            plan = sample.extra['_native_opd_plan']
            stats += torch.tensor([
                1, plan['format_valid'], plan['ct_parse_success'], plan['answer_parse_success'],
                plan['xray_length'], plan['ct_length'], plan['answer_length'], bool(plan['groups']),
            ], dtype=stats.dtype, device=device)
        stats = self.accelerator.reduce(stats, reduction='sum')
        names = ['format_valid_rate', 'ct_parse_success_rate', 'answer_parse_success_rate',
                 'mean_xray_length', 'mean_ct_length', 'mean_answer_length', 'semantic_alignment_success_rate']
        mode = 'train' if self.model.training else 'eval'
        for name, value in zip(names, stats[1:] / stats[0].clamp(min=1)):
            self._metrics[mode][name].append(value.item())
        if self.opdvr_enabled:
            verifier_stats = torch.tensor(
                [sum(opdvr_closed_flags), sum(opdvr_correct_flags)],
                dtype=torch.float64, device=device)
            verifier_stats = self.accelerator.reduce(verifier_stats, reduction='sum')
            self._metrics[mode]['opdvr_closed_rows'].append(verifier_stats[0].item())
            self._metrics[mode]['opdvr_closed_accuracy'].append(
                (verifier_stats[1] / verifier_stats[0].clamp(min=1)).item())

    def _sft_anchor_loss(self, model):
        encoded_rows = to_device(next(self.chord_sft_iterator), 'cpu')
        anchors = []
        for encoded in encoded_rows:
            encoded = dict(encoded)
            labels = [int(t) for t in encoded['labels']]
            positions = [i for i, token in enumerate(labels) if token != -100]
            response_ids = [labels[i] for i in positions]
            mask = sft_anchor_mask(response_ids, self.processing_class)
            for index, active in zip(positions, mask):
                if not active:
                    labels[index] = -100
            encoded['labels'] = labels
            encoded.pop('_extra_kwargs', None)
            anchors.append(GRPOSample(messages=[], encoded=encoded))
        inputs, batch = collate_to_grpo_micro_batch(
            anchors, self.template, device=self.accelerator.device,
            use_logits_to_keep=True, include_routed_masks=False)
        inputs.pop('labels', None)
        # SFT is ordinary temperature=1 cross entropy. The OPD/student rollout
        # scoring temperature is restored afterwards; request_config is untouched.
        temperature = self.temperature
        try:
            self.temperature = 1.0
            logps = self._get_per_token_logps_and_entropies(model, inputs, batch)[0]
        finally:
            self.temperature = temperature
        return -(logps * batch.completion_mask).sum() / batch.completion_mask.sum().clamp(min=1)

    def _compute_loss_and_metrics(self, model, model_inputs, grpo_batch):
        student_logps, _ = self._get_per_token_logps_and_entropies(model, model_inputs, grpo_batch)
        completion_mask = getattr(grpo_batch, 'native_loss_completion_mask', grpo_batch.completion_mask)
        teacher_opd_mask = grpo_batch.teacher_token_mask & completion_mask
        xray_ref_kl_mask = grpo_batch.grpo_token_mask & completion_mask
        grpo_mask = completion_mask if self.grpo_scope == 'full' else xray_ref_kl_mask
        if self.grpo_only:
            loss_teacher_opd = student_logps.sum() * 0
            delta = torch.zeros_like(student_logps)
            loss_xray_ref_kl = xray_reference_kl_loss(
                student_logps, grpo_batch.ref_per_token_logps, completion_mask, xray_ref_kl_mask)
        else:
            answer_mask = getattr(grpo_batch, 'native_answer_content_mask',
                                  torch.zeros_like(completion_mask)) & teacher_opd_mask
            token_weights = None
            if self.teacher_answer_weight != 1.0:
                token_weights = 1.0 + (self.teacher_answer_weight - 1.0) * answer_mask.float()
            loss_teacher_opd, loss_xray_ref_kl, delta = native_opd_losses(
                student_logps, grpo_batch.old_per_token_logps, grpo_batch.teacher_per_token_logps,
                grpo_batch.ref_per_token_logps, completion_mask,
                teacher_opd_mask, xray_ref_kl_mask, grpo_batch.native_student_groups,
                teacher_coefficient=self.teacher_kl_coef,
                epsilon_low=self.epsilon_low, epsilon_high=self.epsilon_high,
                teacher_token_weights=token_weights,
                opdvr_correctness=getattr(grpo_batch, 'native_opdvr_correctness', None),
                opdvr_closed_mask=getattr(grpo_batch, 'native_opdvr_closed_mask', None))
        loss_sft_anchor = self._sft_anchor_loss(model) * getattr(grpo_batch, 'native_active_fraction', 1.0)
        loss_xray_opd = student_logps.sum() * 0
        if self.xray_opd_coefficient:
            xm = grpo_batch.xray_teacher_token_mask & completion_mask
            # Mapping accumulates teacher block probabilities in FP32; use the
            # same dtype for both sides of this additional loss (student scoring
            # may return BF16). The cast preserves gradients to student logps.
            xray_student_logps = student_logps.float()
            xray_old_logps = (grpo_batch.old_per_token_logps.float()
                             if grpo_batch.old_per_token_logps is not None else None)
            loss_xray_opd, _, xd = native_opd_losses(
                xray_student_logps, xray_old_logps,
                grpo_batch.xray_teacher_per_token_logps, xray_student_logps.detach(),
                completion_mask, xm, torch.zeros_like(xm),
                grpo_batch.xray_teacher_groups, teacher_coefficient=self.xray_opd_coefficient,
                epsilon_low=self.epsilon_low, epsilon_high=self.epsilon_high)
            xv = self.accelerator.reduce(torch.stack([
                loss_xray_opd.detach(), (xd.detach() * xm).sum(), xm.sum(),
                ((xm.sum(-1) > 0).sum()), torch.tensor(xm.shape[0], device=xm.device),
            ]).to(torch.float64), reduction='sum')
            mode = 'train' if self.model.training else 'eval'
            self._metrics[mode]['loss_xray_opd'].append((xv[0] / self.accelerator.num_processes).item())
            self._metrics[mode]['xray_opd_token_count'].append(xv[2].item())
            self._metrics[mode]['mean_xray_teacher_student_logprob_delta'].append(
                (xv[1] / xv[2]).item() if xv[2] else float('nan'))
            self._metrics[mode]['xray_opd_alignment_rate'].append((xv[3] / xv[4].clamp(min=1)).item())
            self._metrics[mode]['xray_opd_coefficient'].append(self.xray_opd_coefficient)
        loss_grpo = student_logps.sum() * 0
        if self.xray_grpo_coefficient:
            loss_grpo = xray_grpo_loss(
                student_logps, grpo_batch.old_per_token_logps, grpo_batch.advantages,
                completion_mask, grpo_mask,
                epsilon_low=self.epsilon_low, epsilon_high=self.epsilon_high)
        loss_total = (1.0 * loss_teacher_opd
                      + loss_xray_opd
                      + self.beta * loss_xray_ref_kl
                      + self.sft_anchor_coefficient * loss_sft_anchor
                      + self.xray_grpo_coefficient * loss_grpo)

        values = torch.stack([
            loss_teacher_opd.detach(), loss_xray_ref_kl.detach(), loss_sft_anchor.detach(),
            (delta.detach() * teacher_opd_mask).sum(), teacher_opd_mask.sum(),
            loss_grpo.detach(), xray_ref_kl_mask.sum(), grpo_mask.sum(), grpo_batch.completion_mask.sum(),
        ]).to(torch.float64)
        values = self.accelerator.reduce(values, reduction='sum')
        mode = 'train' if self.model.training else 'eval'
        active_rows = completion_mask.any(-1).float().sum()
        self._metrics[mode]['native_active_rows'].append(
            self.accelerator.reduce(active_rows, reduction='sum').item())
        self._metrics[mode]['rollout_temperature'].append(float(self.request_config.temperature))
        self._metrics[mode]['logprob_temperature'].append(float(self.temperature))
        self._metrics[mode]['rollouts_per_prompt'].append(self.num_generations)
        self._metrics[mode]['grpo_only_ablation'].append(float(self.grpo_only))
        self._metrics[mode]['reward_guided_opd'].append(float(self.reward_guided_opd))
        self._metrics[mode]['opdvr_enabled'].append(float(self.opdvr_enabled))
        if self.opdvr_enabled:
            closed = grpo_batch.native_opdvr_closed_mask[:, None]
            gated_delta = opdvr_gate_delta(
                delta.detach(), grpo_batch.native_opdvr_correctness,
                grpo_batch.native_opdvr_closed_mask)
            closed_tokens = teacher_opd_mask & closed
            nonzero_closed = closed_tokens & (delta.detach() != 0)
            zeroed = nonzero_closed & (gated_delta == 0)
            gate_stats = self.accelerator.reduce(torch.stack([
                zeroed.sum().to(torch.float64), nonzero_closed.sum().to(torch.float64),
            ]), reduction='sum')
            self._metrics[mode]['opdvr_zeroed_nonzero_token_fraction'].append(
                (gate_stats[0] / gate_stats[1].clamp(min=1)).item())
        for name, value in zip(['loss_teacher_opd', 'loss_xray_ref_kl', 'loss_sft_anchor'],
                               values[:3] / self.accelerator.num_processes):
            self._metrics[mode][name].append(value.item())
        # Undefined, not zero, when no semantic content received OPD.
        mean_delta = (values[3] / values[4]).item() if values[4] else float('nan')
        self._metrics[mode]['mean_teacher_student_logprob_delta'].append(mean_delta)
        self._metrics[mode]['teacher_opd_token_count'].append(values[4].item())
        answer_tokens = (getattr(grpo_batch, 'native_answer_content_mask',
                                 torch.zeros_like(completion_mask)) & teacher_opd_mask).sum()
        self._metrics[mode]['teacher_answer_token_count'].append(
            self.accelerator.reduce(answer_tokens, reduction='sum').item())
        self._metrics[mode]['teacher_answer_weight'].append(self.teacher_answer_weight)
        self._metrics[mode][f'loss_{self.grpo_scope}_grpo'].append(
            (values[5] / self.accelerator.num_processes).item())
        self._metrics[mode]['xray_body_token_count'].append(values[6].item())
        self._metrics[mode]['xray_grpo_coefficient'].append(self.xray_grpo_coefficient)
        self._metrics[mode]['grpo_scope_full'].append(float(self.grpo_scope == 'full'))
        self._metrics[mode]['grpo_token_count'].append(values[7].item())
        self._metrics[mode]['grpo_completion_coverage'].append((values[7] / values[8].clamp(min=1)).item())
        return loss_total, {'mode': mode, 'entropy': {}}
