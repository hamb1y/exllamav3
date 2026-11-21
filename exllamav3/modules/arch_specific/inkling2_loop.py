"""Text decoder math for the fixed-pass Inkling 2 Mini checkpoint."""

import torch
import torch.nn.functional as F

from ..linear import Linear
from ..module import Module
from ..quant import LinearFP16
from ...util.device_copy import to_device


def rms(x, eps, weight = None):
    x = x.float()
    x = x * (x.square().mean(-1, keepdim = True) + eps).rsqrt()
    return x if weight is None else x * weight.float()


def softcap(x, cap):
    return cap * (x / cap).tanh()


def clip(x, threshold, scale = None):
    x = x.float()
    if scale is not None:
        x = x * scale.float()
    return x * (threshold / (x.norm(dim = -1, keepdim = True) + 1e-6)).clamp(max = 1)


def mix(blocks, x, queries, eps):
    states = rms(torch.stack((*blocks, x), dim = -2), eps)
    queries = queries.reshape(-1, x.shape[-1]).float()
    scores = (states @ queries.T) * (x.shape[-1] ** -0.5)
    return (scores.softmax(dim = -2).transpose(-1, -2) @ states).unbind(dim = -2)


def control(x, message, weights, eps):
    down, forget, gain = weights
    gate_input = torch.cat((rms(x, eps), rms(message, eps)), dim = -1)
    gate = F.linear(gate_input.to(down.dtype), down).float()
    return x * (1 - F.linear(gate, forget.float()).tanh()) + message * F.linear(gate, gain.float()).tanh()


def convolve(x, weight, taps = None):
    # Standard conv weights are oldest-first; dynamic kernels and taps are lag-first.
    dims = x.shape
    x = x.float()
    k = weight.shape[-1]
    history = F.pad(x.transpose(1, 2), (k - 1, 0))
    y = F.conv1d(history, weight.reshape(dims[-1], 1, k).float(), groups = dims[-1]).transpose(1, 2)
    if taps is not None:
        heads = taps.shape[-1] // k
        hist = history.unfold(-1, k, 1).permute(0, 2, 1, 3)
        hist = hist.reshape(*dims[:2], heads, -1, k)
        y = y + (hist * taps.float().reshape(*dims[:2], heads, 1, k).flip((-1,))).sum(-1).reshape(dims)
    return (x + y) * (((2 if taps is not None else 1) * k + 1) ** -0.5)


def relative_attention(q, k, v, table, sinks, window):
    """Causal GQA with relative logit bias and a zero-valued sink for each query."""
    batch, length, heads, d = q.shape
    kv_heads = k.shape[-2]
    groups = heads // kv_heads
    extent = table.shape[-1]
    out = torch.empty_like(q)
    # Bound score and bias intermediates, including during calibration.
    max_keys = min(length, window + 128) if window else length
    chunk = max(1, min(128, (1 << 22) // (batch * heads * max_keys)))
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        lo = max(0, start - window + 1) if window else 0
        dist = (torch.arange(start, end, device = q.device)[:, None] -
                torch.arange(lo, end, device = q.device)[None, :])
        x = q[:, start:end].reshape(batch, end - start, kv_heads, groups, d)
        x = x.permute(0, 2, 3, 1, 4).reshape(batch, kv_heads, -1, d)
        scores = (x @ k[:, lo:end].permute(0, 2, 3, 1)) * (d ** -0.5)
        scores = scores.reshape(batch, heads, end - start, end - lo)
        bias = table[:, start:end].transpose(1, 2).gather(
            -1, dist.clamp(0, extent - 1).expand(batch, heads, -1, -1)
        )
        scores += bias.masked_fill((dist < 0) | (dist >= extent), 0)
        mask = dist < 0
        if window:
            mask = mask | (dist >= window)
        scores.masked_fill_(mask, -torch.inf)
        sink = sinks[:, start:end].transpose(1, 2).unsqueeze(-1)
        prob = torch.cat((scores, sink), dim = -1).softmax(dim = -1)[..., :-1]
        prob = prob.reshape(batch, kv_heads, groups * (end - start), end - lo)
        y = prob @ v[:, lo:end].transpose(1, 2)
        out[:, start:end] = y.reshape(batch, heads, end - start, d).transpose(1, 2)
    return out


def unpack_expert(weight, scale, outer = None):
    if outer is None:
        # MXFP8 stores E4M3 weights and unsigned E8M0 scales (one scale per 32 inputs).
        block = 32
        weight = weight.view(torch.float8_e4m3fn).float()
        scale = torch.exp2(scale.view(torch.uint8).float() - 127)
    else:
        # NVFP4 uses two E2M1 values per byte, least-significant nibble first.
        block = 16
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                           device = weight.device)
        weight = torch.stack((lut[(weight & 15).long()], lut[(weight >> 4).long()]), dim = -1).flatten(-2)
        scale = scale.view(torch.float8_e4m3fn).float()
    weight = (weight.unflatten(-1, (-1, block)) * scale.unsqueeze(-1)).flatten(-2)
    return weight if outer is None else weight * outer


class Inkling2ExpertLinear(Linear):
    """Read one expert from the original packed checkpoint, or use normal EXL3 tensors."""

    def __init__(self, config, key, in_features, out_features, qmap, packed_key, expert, row_parity = None):
        super().__init__(config, key, in_features, out_features, qmap = qmap, trim_padded_out = True)
        self.packed_key = packed_key
        self.expert = expert
        self.row_parity = row_parity

    def can_defer_load(self):
        return not self.config.stc.has_tensor(self.packed_key) and super().can_defer_load()

    def load_fp16(self, key):
        if super().load_fp16(key):
            return True
        stc, p = self.config.stc, self.packed_key
        if not stc.has_tensor(p):
            return False
        kwargs = dict(fidx = self.expert, no_defer = True, allow_bf16 = True, arena = False)
        weight = stc.get_tensor(p, self.device, **kwargs)
        scale = stc.get_tensor(p + "_scale", self.device, **kwargs)
        outer = None
        if stc.has_tensor(p + "_amax"):
            amax = stc.get_tensor(p + "_amax", "cpu", no_defer = True)[self.expert]
            denom = stc.get_tensor(p + "_outer_scale_denominator", "cpu", no_defer = True)
            outer = (amax / denom).item()
        if self.row_parity is not None:
            weight = weight[self.row_parity::2]
            scale = scale[self.row_parity::2]
        weight = unpack_expert(weight, scale, outer).to(torch.half).T.contiguous()
        self.inner = LinearFP16(
            self.in_features, self.out_features, self.pad_out(weight), None,
            self.full_in_features, self.full_out_features, self.first_in_feature, self.first_out_feature,
            key = self.key,
        )
        self.quant_type = "fp16"
        return True


class Inkling2Layer(Module):

    def __init__(self, config, index):
        key = f"model.llm.layers.{index}"
        super().__init__(config, key, None)
        self.index = index
        self.local = index in config.local_layers
        self.heads, self.kv_heads, self.head_dim = config.attention_dims(self.local)
        d, h, kv, dh = config.hidden_size, self.heads, self.kv_heads, self.head_dim
        self.in_core = config.core_start <= index < config.core_end
        self.boundary = index % config.block_size == 0
        self.statics = {}
        self.attn_query = self.mlp_query = self.message_query = None

        self.proj = {}
        for name, ni, no, group in [
            ("wq_du", d, h * dh, "query"), ("wk_dv", d, kv * dh, "key"),
            ("wv_dv", d, kv * dh, "value"), ("wr_du", d, h * config.d_rel, "query"),
            ("gate_attn", d, h * dh, "query"), ("wo_ud", h * dh, d, "output"),
        ]:
            layer = Linear(config, f"{key}.attn.{name}", ni, no,
                           qmap = f"{key}.{group}", trim_padded_out = True)
            self.proj[name] = layer
            self.register_submodule(layer)

        self.experts = []
        p = key + ".mlp.experts"
        inter = config.moe_intermediate_size
        for i in range(config.num_experts):
            gate = Inkling2ExpertLinear(config, f"{p}.{i}.gate_proj", d, inter, p + ".in", p + ".w13_weight", i, 0)
            up = Inkling2ExpertLinear(config, f"{p}.{i}.up_proj", d, inter, p + ".in", p + ".w13_weight", i, 1)
            down = Inkling2ExpertLinear(config, f"{p}.{i}.down_proj", inter, d, f"{p}.{i}.down",
                                       p + ".w2_weight", i)
            self.experts.append((gate, up, down))
            self.modules.extend((gate, up, down))

    def can_defer_load(self):
        return False

    def load(self, device, **kwargs):
        raw = kwargs.pop("raw_stc", self.config.stc)
        super().load(device, **kwargs)

        def read(stc, k):
            return stc.get_tensor(k, device, allow_bf16 = True, no_defer = True)

        names = [
            "attn.attn_sink_proj.weight", "attn.q_l2_clip.scale", "attn.rel_l2_clip.scale",
            "attn.rel_logits_proj.proj", "attn_sconv.weight", "mlp_sconv.weight", "mlp_norm.weight",
            "mlp.gate.weight", "mlp.gate.bias",
        ]
        if self.local:
            names += ["attn_norm.weight", "attn.k_sconv.weight", "attn.v_sconv.weight"]
        else:
            names += [f"{kind}_input_norm.weight" for kind in "qkv"]
            names += [f"attn.{kind}_sconv.{name}" for kind in "kv" for name in ("static_w", "weight_proj")]
        if self.in_core:
            names += [f"controllers.{kind}.{name}_weight" for kind in ("attn", "k", "v", "mlp")
                      for name in ("down", "forget", "gain")]

        self.statics = {k: read(self.config.stc, f"{self.key}.{k}") for k in names}
        prefix = "model.llm"
        self.attn_query = read(raw, f"{prefix}.attn_res.attn_queries.{self.index - 1}") if self.index else None
        self.mlp_query = read(raw, f"{prefix}.attn_res.mlp_queries.{self.index}")
        if self.in_core:
            i = (self.index - self.config.core_start) * 2
            q = read(raw, f"{prefix}.middle_cycle_cross_pass_queries")
            kv = read(raw, f"{prefix}.middle_cycle_cross_pass_kv_queries")
            self.message_query = torch.stack((q[i], kv[i], kv[i + 1], q[i + 1]))

    def unload(self):
        super().unload()
        self.statics.clear()
        self.attn_query = self.mlp_query = self.message_query = None

    def get_tensors(self):
        return {f"{self.key}.{k}": v.cpu() for k, v in self.statics.items()}

    def weights_numel(self):
        return super().weights_numel() + sum(w.numel() for w in self.statics.values())

    def optimizer_targets(self):
        return [k for linear in self.modules for k in linear.optimizer_targets()]

    def attention(self, xq, xk, xv, params):
        c = self.config
        h, kv, dh = self.heads, self.kv_heads, self.head_dim
        w = self.statics

        def proj(name, x):
            return self.proj[name].forward(x.contiguous(), params, out_dtype = torch.float)

        def conv_kv(kind, x):
            y = proj(f"w{kind}_dv", x).to(torch.bfloat16)
            p = f"attn.{kind}_sconv."
            if self.local:
                return convolve(y, w[p + "weight"])
            taps = F.linear(x.float(), w[p + "weight_proj"].float())
            return convolve(y, w[p + "static_w"].T.flip((-1,)), taps)

        q = clip(proj("wq_du", xq).unflatten(-1, (h, dh)), c.qk_clip, w["attn.q_l2_clip.scale"])
        k = clip(conv_kv("k", xk).unflatten(-1, (kv, dh)), c.qk_clip)
        v = rms(conv_kv("v", xv).unflatten(-1, (kv, dh)), c.rms_norm_eps)
        r = clip(proj("wr_du", xq).unflatten(-1, (h, c.d_rel)), c.rel_clip, w["attn.rel_l2_clip.scale"])
        if not self.local:
            pos = torch.arange(1, q.shape[1] + 1, device = q.device, dtype = torch.float)
            tau = 1 + c.log_alpha * (pos / c.log_floor).clamp(min = 1).log()
            q = q * tau[None, :, None, None]
            r = r * tau[None, :, None, None]
        table = r @ w["attn.rel_logits_proj.proj"].float()
        sinks = F.linear(xq.float(), w["attn.attn_sink_proj.weight"].float())
        y = relative_attention(q, k, v, table, sinks, c.window if self.local else None).flatten(-2)
        y = (y * proj("gate_attn", xq).sigmoid()).to(torch.half)
        return convolve(proj("wo_ud", y).to(torch.bfloat16), w["attn_sconv.weight"])

    def moe(self, x, params):
        c, w = self.config, self.statics
        flat = x.reshape(-1, x.shape[-1])
        # All gate/up projections share the same input Hessian. Capture all tokens
        # once, not just the tokens routed to whichever expert is visited first.
        first_gate = self.experts[0][0]
        gate_params = params
        if "capture" in params:
            if params.get("capture_qmaps") is None or first_gate.qmap in params["capture_qmaps"]:
                first_gate.capture_H(flat, params)
            gate_params = {k: v for k, v in params.items() if k != "capture"}
        scores = F.linear(flat.float(), w["mlp.gate.weight"].float()).sigmoid()
        chosen = (scores + w["mlp.gate.bias"].float()).topk(c.num_experts_per_tok, sorted = False).indices
        weights = scores.gather(-1, chosen)
        weights = weights / (weights.sum(-1, keepdim = True) + 1e-20) * c.route_scale
        out = torch.zeros_like(flat, dtype = torch.float)
        # One host transfer per layer, rather than synchronizing once for each expert.
        order = chosen.flatten().argsort()
        counts = chosen.flatten().bincount(minlength = c.num_experts).tolist()
        offset = 0
        flat_weights = weights.flatten()
        for i, (gate, up, down) in enumerate(self.experts):
            count = counts[i]
            if count == 0:
                # A short calibration row might never select some experts. Give the
                # missing down projection a small activation sample without affecting
                # the actual routed result or the shared input Hessian.
                qmaps = params.get("capture_qmaps")
                if "capture" in params and down.qmap not in params["capture"] and (qmaps is None or down.qmap in qmaps):
                    inputs = flat[:min(32, len(flat))]
                    a = self.expert_activation(gate, up, inputs, gate_params)
                    down.capture_H(a, params)
                continue
            assignment = order[offset:offset + count]
            offset += count
            token = assignment // c.num_experts_per_tok
            inputs = flat.index_select(0, token)
            a = self.expert_activation(gate, up, inputs, gate_params)
            y = down.forward(a, params, out_dtype = torch.float)
            out.index_add_(0, token, y * flat_weights[assignment, None])
        return out.view_as(x)

    def expert_activation(self, gate, up, inputs, params):
        g = gate.forward(inputs, params, out_dtype = torch.float)
        u = up.forward(inputs, params, out_dtype = torch.float)
        return (softcap(g, self.config.gate_cap) * g.sigmoid() * softcap(u, self.config.up_cap)).to(torch.half)

    def forward(self, x, params, out_dtype = None):
        context = params.get("quant_preserve", params)
        stored = context.get("inkling_blocks", ())
        blocks = tuple(to_device(t, x.device) for t in stored)
        previous = context.get("inkling_previous")
        c, w = self.config, self.statics
        eps = c.rms_norm_eps
        messages = None
        if self.in_core and previous is not None:
            bank, last = previous
            messages = mix(tuple(to_device(t, x.device) for t in bank),
                           to_device(last, x.device), self.message_query, eps)

        xqkv = (x,) if self.attn_query is None else mix(blocks, x, self.attn_query, eps)
        if len(xqkv) == 1:
            xqkv *= 3
        norm_names = ["attn_norm.weight"] * 3 if self.local else [f"{k}_input_norm.weight" for k in "qkv"]
        if messages is not None:
            xqkv = tuple(self.control(kind, y, msg) for kind, y, msg in zip(("attn", "k", "v"), xqkv, messages))
        inputs = [rms(y, eps, w[n]).to(torch.bfloat16).to(torch.half) for y, n in zip(xqkv, norm_names)]
        a = self.attention(*inputs, params).to(torch.bfloat16).float()
        if self.boundary:
            context["inkling_blocks"] = (*stored, x.cpu() if "quant_preserve" in params else x)
            blocks = (*blocks, x)
        else:
            a = a + x.float()
        (m,) = mix(blocks, a, self.mlp_query, eps)
        if messages is not None:
            m = self.control("mlp", m, messages[3])
        m = self.moe(rms(m, eps, w["mlp_norm.weight"]).to(torch.bfloat16).to(torch.half), params)
        m = convolve(m.to(torch.bfloat16), w["mlp_sconv.weight"]).to(torch.bfloat16)
        return (a + m).to(torch.bfloat16)

    def control(self, kind, x, msg):
        weights = [self.statics[f"controllers.{kind}.{k}_weight"] for k in ("down", "forget", "gain")]
        return control(x.float(), msg, weights, self.config.rms_norm_eps)
