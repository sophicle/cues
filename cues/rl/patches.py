"""Four patches to TRL 1.10's GRPOTrainer. Each is gradient-neutral or restores the documented
loss; together they make the backward ~3x faster on long-tailed completions.

1. TRIM. TRL pads every completion in a generation batch to the batch's longest sequence and
   never trims again, so with a 4,096 cap the backward runs at the cap essentially always
   (median completion is a few hundred tokens). Cut each microbatch's completion axis to its own
   longest unmasked position. Removed positions have completion_mask == 0, which is both the loss
   mask and half of the attention mask, and attention is causal, so they contribute nothing.

2. SORT. Sort the generation batch by length before it is split into accumulation microbatches,
   instead of TRL's shuffle, so the trim has less to pad. Only sound if the loss normaliser is
   the same constant for every microbatch, which (3) guarantees.

3. GLOBAL NORMALISER. loss_type="dapo" documents a token-level loss normalised by the token
   count of the whole generation batch. With use_liger_kernel=True, TRL does not pass
   num_items_in_batch into the Liger loss, and Liger falls back to normalising each microbatch by
   its own token count, which weights tokens by the length mix of the microbatch they landed in.
   Pass it through, and multiply the returned loss back by the accumulation steps, because
   compute_liger_loss divides by that count on the assumption of the per-microbatch denominator.
   Without the compensation every gradient shrinks by a factor of accum.

Also: vLLM sleep mode. TRL empties torch's allocator cache before the weight wake-up but not
before the KV-cache wake-up; the weight sync in between can leave the pages gone. Empty before
every wake.

4. PER-MODULE WEIGHT SYNC UNDER ZeRO-3 (--zero3 with LoRA). To push merged weights to vLLM, TRL
   gathers EVERY parameter of the model at once and merges the adapters on the full copy, which
   materialises the whole base model on every GPU (64 GB for a 32B) and OOMs on 80 GB cards.
   Gather one LoRA-wrapped layer at a time instead: merge it, push its base weight, unmerge,
   release. Peak extra memory is one layer's weights. Same names, same values as TRL's path.

Provenance: avdravid/reasoning_registers_grpo GRPO_clean src/patches.py.
"""
import torch


def apply(liger=True):
    import vllm
    from trl import GRPOTrainer
    from trl.trainer import grpo_trainer as gt

    _wake = vllm.LLM.wake_up

    def wake_with_clean_cache(self, *a, **k):
        torch.cuda.empty_cache()
        return _wake(self, *a, **k)

    vllm.LLM.wake_up = wake_with_clean_cache

    keys = ("completion_ids", "completion_mask", "tool_mask", "old_per_token_logps",
            "ref_per_token_logps", "sampling_per_token_logps", "importance_sampling_ratio")
    _prepare = GRPOTrainer._prepare_inputs

    def prepare_trimmed(self, generation_batch):
        inputs = _prepare(self, generation_batch)
        m = inputs.get("completion_mask")
        if not self.model.training or not torch.is_tensor(m) or m.dim() != 2:
            return inputs
        width = m.size(1)
        keep = max(1, int(m.sum(dim=1).max().item()))
        if keep >= width:
            return inputs
        out = dict(inputs)
        for k in keys:
            v = out.get(k)
            if torch.is_tensor(v) and v.dim() == 2 and v.size(1) == width:
                out[k] = v[:, :keep]
        return out

    GRPOTrainer._prepare_inputs = prepare_trimmed

    _shuffle = gt.shuffle_sequence_dict

    def sort_by_length(seq_dict):
        m = seq_dict.get("completion_mask")
        if not torch.is_tensor(m) or m.dim() != 2:
            return _shuffle(seq_dict)
        order = torch.argsort(m.sum(dim=1), stable=True).tolist()

        def take(v):
            if v is None or (isinstance(v, torch.Tensor) and v.ndim == 0):
                return v
            return v[order] if isinstance(v, torch.Tensor) else [v[i] for i in order]

        return {k: take(v) for k, v in seq_dict.items()}

    gt.shuffle_sequence_dict = sort_by_length

    if not liger:
        return
    _liger = GRPOTrainer.compute_liger_loss

    def liger_global_norm(self, unwrapped_model, inputs):
        orig = self.liger_loss

        class WithNorm:
            def __call__(_s, **kw):
                kw.setdefault("num_items_in_batch", inputs["num_items_in_batch"])
                return orig(**kw)

        self.liger_loss = WithNorm()
        try:
            loss = _liger(self, unwrapped_model, inputs)
        finally:
            self.liger_loss = orig
        return loss * self.current_gradient_accumulation_steps

    GRPOTrainer.compute_liger_loss = liger_global_norm


def apply_zero3_sync():
    """(4) above. Only the PEFT + ZeRO-3 branch of VLLMGeneration.sync_weights changes."""
    from peft.tuners.lora.layer import LoraLayer
    from trl.generation import vllm_generation as vg
    from accelerate.utils import is_peft_model

    cls = next(c for c in vars(vg).values() if isinstance(c, type) and "sync_weights" in vars(c))
    _orig = cls.sync_weights

    def sync_weights(self):
        model = self.model
        if not (is_peft_model(model) and self._dist.is_zero3):
            return _orig(self)
        if self.mode == "colocate" and self.enable_sleep_mode:
            torch.cuda.empty_cache()
            self.llm.wake_up(tags=["weights"])
            self._llm_weights_sleeping = False

        def vllm_name(n):
            n = n.removeprefix("base_model.model.").replace(".base_layer", "")
            return self._fix_param_name_to_vllm(n, extra_prefixes=["modules_to_save.default."])

        done = set()
        for mname, module in model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            with self._dist.gather_params([p for _, p in module.named_parameters()]):
                module.merge()
                for pn, p in module.get_base_layer().named_parameters(recurse=False):
                    self._push_param_to_vllm(vllm_name(f"{mname}.base_layer.{pn}"), p.data)
                module.unmerge()
            done.update(f"{mname}.{pn}" for pn, _ in module.named_parameters())
        for name, param in model.named_parameters():
            if name in done or model.prefix in name or "original_module" in name:
                continue
            with self._dist.gather_params([param]):
                self._push_param_to_vllm(vllm_name(name), param.data)
        if self.mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.mode == "colocate":
            self.llm.reset_prefix_cache()

    cls.sync_weights = sync_weights
