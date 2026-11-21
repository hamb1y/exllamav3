from copy import copy
from pathlib import Path

import torch

from ..loader.safetensors import SafetensorsCollection
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Embedding, Linear, Module
from ..modules.arch_specific.inkling2_loop import Inkling2Layer, mix, rms
from ..util.device_copy import to_device


class Inkling2LoopConfig(Config):
    arch_string = "Inkling2LoopForConditionalGeneration"

    def __init__(self, directory, **kwargs):
        super().__init__(directory, {"text": Inkling2LoopModel}, config_key = "text_config", **kwargs)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", no_default)
        self.head_dim = self.read_cfg(int, "head_dim", no_default)
        self.swa_dims = (
            self.read_cfg(int, "swa_num_attention_heads", self.num_q_heads),
            self.read_cfg(int, "swa_num_key_value_heads", self.num_kv_heads),
            self.read_cfg(int, "swa_head_dim", self.head_dim),
        )
        self.d_rel = self.read_cfg(int, "d_rel", no_default)
        self.local_layers = frozenset(self.read_cfg(list, "local_layer_ids", no_default))
        self.window = self.read_cfg(int, "sliding_window_size", no_default)
        self.block_size = self.read_cfg(int, "n_layers_per_attn_res_block", no_default)
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)
        self.qk_clip = self.read_cfg(float, "qk_l2_clip_threshold", no_default)
        self.rel_clip = self.read_cfg(float, "rel_l2_clip_threshold", no_default)
        self.log_floor = self.read_cfg(int, "log_scaling_n_floor", no_default)
        self.log_alpha = self.read_cfg(float, "log_scaling_alpha", no_default)
        self.num_experts = self.read_cfg(int, "n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.gate_cap = self.read_cfg(float, "mlp_gate_softcap", no_default)
        self.up_cap = self.read_cfg(float, "mlp_up_softcap", no_default)
        self.route_scale = self.read_cfg(float, "route_scale", no_default)
        self.logit_cap = self.read_cfg(float, "final_logit_softcapping", no_default)
        self.padded_vocab_size = self.vocab_size
        self.vocab_size = self.read_cfg(int, "unpadded_vocab_size", self.padded_vocab_size)
        self.max_position_embeddings = self.read_cfg(int, "model_max_length", self.max_position_embeddings)

        loop = self.read_cfg(dict, "loop_config", no_default)
        self.core_start, self.core_end = loop["core_start"], loop["core_end"]
        self.num_passes = loop["num_passes"]
        if (
            loop.get("is_varloop", False) or self.num_passes < 2 or loop["gate_rank"] < 1
            or not (0 < self.core_start < self.core_end <= self.num_hidden_layers)
            or self.block_size < 1 or self.core_start % self.block_size or self.core_end % self.block_size
        ):
            raise ValueError("Unsupported Inkling 2 loop configuration (requires fixed, block-aligned passes)")
        if self.layer_map or self.layer_map_str:
            raise ValueError("Inkling 2 Mini requires its trained layer order; custom layer maps are unsupported")
        if (
            self.read_cfg(bool, "q_bias", False) or self.read_cfg(bool, "o_bias", False)
            or not self.read_cfg(bool, "use_gate_bias", True)
            or self.read_cfg(int, "n_shared_experts", 0)
            or not self.read_cfg(bool, "attn_res_normalize_values", True)
            or not self.read_cfg(bool, "use_value_norm_no_gamma", True)
        ):
            raise ValueError("Unsupported Inkling 2 Mini attention or router configuration")

    def attention_dims(self, local):
        return self.swa_dims if local else (self.num_q_heads, self.num_kv_heads, self.head_dim)


class Inkling2Norm(Module):

    def __init__(self, config, key, final = False):
        super().__init__(config, key, None)
        self.final = final
        self.weight = self.query = None

    def can_defer_load(self):
        return False

    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.weight = stc.get_tensor(self.key + ".weight", device, allow_bf16 = True, no_defer = True)
        if self.final:
            self.query = stc.get_tensor("model.llm.attn_res.output_query", device, allow_bf16 = True, no_defer = True)

    def unload(self):
        super().unload()
        self.weight = self.query = None

    def get_tensors(self):
        return {self.key + ".weight": self.weight.cpu()}

    def weights_numel(self):
        return self.config.hidden_size

    def optimizer_targets(self):
        return []

    def forward(self, x, params, out_dtype = None):
        ctx = params.get("quant_preserve", params)
        eps = self.config.rms_norm_eps
        if self.final:
            blocks = ctx.pop("inkling_blocks")
            (x,) = mix(tuple(to_device(block, x.device) for block in blocks), x, self.query, eps)
            ctx.pop("inkling_previous", None)
            ctx.pop("inkling_prelude", None)
        return rms(x, eps, self.weight).to(torch.bfloat16).to(torch.half if self.final else torch.bfloat16)


class Inkling2PassStart(Module):
    """The first core pass records its inputs; later passes restart from them."""

    def __init__(self, config, number):
        super().__init__(config, f"inkling_pass_start.{number}", None)
        self.number = number

    def forward(self, x, params, out_dtype = None):
        ctx = params.get("quant_preserve", params)
        blocks = ctx.get("inkling_blocks", ())
        saved = x.cpu() if "quant_preserve" in params else x
        if self.number == 0:
            ctx["inkling_prelude"] = (blocks, saved)
            return x
        prelude_blocks, prelude = ctx["inkling_prelude"]
        # The first boundary of the core stores the prelude itself, not a new message.
        ctx["inkling_previous"] = (blocks[len(prelude_blocks) + 1:], saved)
        ctx["inkling_blocks"] = prelude_blocks
        if self.number == self.config.num_passes - 1:
            del ctx["inkling_prelude"]
        return to_device(prelude, x.device)

    def optimizer_targets(self):
        return []


class Inkling2Replay(Module):
    """Execute a physical layer again without exposing a second set of quantization targets."""

    def __init__(self, config, number, layer):
        super().__init__(config, f"inkling_replay.{number}.{layer.index}", None)
        self.layer = layer
        self.staged_layer = None
        self.quant_dir = None

    def can_defer_load(self):
        return False

    def load(self, device, **kwargs):
        if self.quant_dir is None:
            # At inference, the physical layer owns its placement and weights.
            self.device = self.layer.device
            return
        # The converter unloads every physical layer after writing it. Reopen only
        # this layer's converted weights, also when resuming from a checkpoint.
        self.device = device
        config = copy(self.config)
        config.stc = SafetensorsCollection(str(self.quant_dir / (self.layer.key + ".safetensors")))
        self.staged_layer = Inkling2Layer(config, self.layer.index)
        try:
            self.staged_layer.load(device, raw_stc = self.config.stc)
        finally:
            config.stc.close()

    def unload(self):
        super().unload()
        if self.staged_layer is not None:
            self.staged_layer.unload()
            self.staged_layer = None

    def forward(self, x, params, out_dtype = None):
        layer = self.staged_layer if self.staged_layer is not None else self.layer
        return layer.forward(x, params, out_dtype)

    def optimizer_targets(self):
        return []


class Inkling2Head(Linear):

    def forward(self, x, params, out_dtype = None):
        logits = super().forward(x, params, out_dtype)
        return logits[..., :self.config.vocab_size].contiguous()


class Inkling2LoopModel(Model):
    config_class = Inkling2LoopConfig

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # A reference text backend, without paged KV/conv state or graph capture.
        self.caps.update({"supports_tp": False, "autosplit_load_fwd": False, "serial_calibration": True})
        self.modules = [
            Embedding(config, "model.llm.embed", vocab_size = config.padded_vocab_size,
                      hidden_size = config.hidden_size),
            Inkling2Norm(config, "model.llm.embed_norm"),
        ]
        self.first_block_idx = len(self.modules)
        layers = [Inkling2Layer(config, i) for i in range(config.num_hidden_layers)]
        start, end = config.core_start, config.core_end
        self.modules += layers[:start] + [Inkling2PassStart(config, 0)] + layers[start:end]
        for number in range(1, config.num_passes):
            self.modules.append(Inkling2PassStart(config, number))
            self.modules += [Inkling2Replay(config, number, layer) for layer in layers[start:end]]
        self.modules += layers[end:]
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [
            Inkling2Norm(config, "model.llm.norm", final = True),
            Inkling2Head(config, "model.llm.unembed", config.hidden_size, config.padded_vocab_size,
                         qmap = "head", qbits_key = "head_bits", caps = {"logits_output": True},
                         softcap = config.logit_cap, out_dtype = torch.float),
        ]
        self.logit_layer_idx = len(self.modules) - 1

    def prepare_conversion(self, work_dir):
        for module in self.modules:
            if isinstance(module, Inkling2Replay):
                module.quant_dir = Path(work_dir) / "qtensors"

    @staticmethod
    def get_additional_compiled_tensors(config):
        stc = config.stc
        tensors = stc.list_tensors("model.llm.attn_res")
        for suffix in ("queries", "kv_queries"):
            tensors.update(stc.list_tensors("model.llm.middle_cycle_cross_pass_" + suffix))
        return tensors

    def prepare_inputs(self, input_ids, params):
        if self.cache_weakrefs or params.get("block_table") is not None:
            raise NotImplementedError("Inkling 2 Mini currently requires uncached text forward passes")
        if params.get("indexed_embeddings") or params.get("embeddings") is not None:
            raise NotImplementedError("Inkling 2 Mini currently supports text inputs only")
        if input_ids.ndim != 2 or min(input_ids.shape) == 0:
            raise ValueError("Inkling 2 Mini expects a non-empty (batch, sequence) token tensor")
        for key in ("inkling_blocks", "inkling_prelude", "inkling_previous"):
            params.pop(key, None)
        return input_ids
