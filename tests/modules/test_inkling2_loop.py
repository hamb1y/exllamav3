import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from exllamav3 import Config, Model
from exllamav3.architecture.inkling2_loop import Inkling2PassStart, Inkling2Replay
from exllamav3.modules.arch_specific.inkling2_loop import convolve, relative_attention, unpack_expert


@pytest.mark.nogpu
def test_packed_expert_formats():
    weight = torch.full((2, 8), 0xA3, dtype = torch.uint8)
    scales = torch.tensor([[2], [3]], dtype = torch.float8_e4m3fn)
    want = torch.tensor([[1.5, -1.] * 8, [2.25, -1.5] * 8])
    torch.testing.assert_close(unpack_expert(weight, scales, 0.5), want)

    weight = torch.full((2, 32), 0.5, dtype = torch.float8_e4m3fn)
    scale = torch.tensor([[128], [126]], dtype = torch.uint8)
    want = torch.tensor([[1.] * 32, [.25] * 32])
    torch.testing.assert_close(unpack_expert(weight, scale), want)


@pytest.mark.nogpu
@pytest.mark.parametrize("window", [None, 3])
def test_relative_attention(window):
    rng = torch.Generator().manual_seed(37)
    q = torch.randn((2, 9, 4, 8), generator = rng)
    k = torch.randn((2, 9, 2, 8), generator = rng)
    v = torch.randn((2, 9, 2, 8), generator = rng)
    table = torch.randn((2, 9, 4, 2), generator = rng)
    sinks = torch.randn((2, 9, 4), generator = rng)
    want = torch.empty_like(q)
    for row in range(q.shape[1]):
        for head in range(q.shape[2]):
            lo = max(0, row - window + 1) if window else 0
            logits = (q[:, row:row + 1, head] @ k[:, lo:row + 1, head // 2].transpose(1, 2))[:, 0] / 8**.5
            for col in range(lo, row + 1):
                if row - col < table.shape[-1]:
                    logits[:, col - lo] += table[:, row, head, row - col]
            probs = torch.cat((logits, sinks[:, row, head, None]), dim = -1).softmax(-1)
            want[:, row, head] = (probs[:, :-1, None] * v[:, lo:row + 1, head // 2]).sum(1)
    torch.testing.assert_close(relative_attention(q, k, v, table, sinks, window), want)


@pytest.mark.nogpu
def test_convolution_tap_order():
    x = torch.tensor([[[1.], [2.], [10.]]])
    w = torch.tensor([[3., 4.]])
    taps = torch.tensor([[[5., 6.], [5., 6.], [5., 6.]]])
    torch.testing.assert_close(convolve(x, w, taps).flatten(), torch.tensor([10., 29., 118.]) / 5**.5)


def write_minimal_checkpoint(path: Path):
    """A three-layer fixed loop with real checkpoint keys and both packed formats."""
    d, inter, vocab, experts, rank = 128, 128, 128, 3, 4
    eps, kernel, heads, kv, dh, d_rel, extent = 1e-6, 2, 4, 2, 32, 8, 4
    t = dict(
        hidden_size = d, vocab_size = vocab, unpadded_vocab_size = vocab - 3, num_hidden_layers = 3,
        num_attention_heads = heads, num_key_value_heads = kv, head_dim = dh, d_rel = d_rel,
        rel_extent = extent, local_layer_ids = [0, 2], sliding_window_size = extent,
        n_layers_per_attn_res_block = 1, sconv_kernel_size = kernel, rms_norm_eps = eps,
        qk_l2_clip_threshold = 12., rel_l2_clip_threshold = 8.,
        log_scaling_n_floor = 2, log_scaling_alpha = .1,
        n_routed_experts = experts, num_experts_per_tok = 2, intermediate_size = inter,
        mlp_gate_softcap = 4., mlp_up_softcap = 25., route_scale = 8., final_logit_softcapping = 30.,
        loop_config = dict(core_start = 1, core_end = 2, num_passes = 3, gate_rank = rank, is_varloop = False),
    )
    cfg = dict(
        architectures = ["Inkling2LoopForConditionalGeneration"], model_type = "inkling2_loop",
        text_config = t, bos_token_id = 2, eos_token_id = 1, pad_token_id = 0,
        quantization_config = dict(quant_method = "modelopt", quant_algo = "NVFP4"),
    )
    (path / "config.json").write_text(json.dumps(cfg))
    vocab_map = {"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3, "e": 4, "Ġ": 5, "t": 6, "Ġt": 7}
    tok = HFTokenizer(BPE(vocab_map, [("Ġ", "t")], unk_token = "[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space = False)
    tok.decoder = ByteLevelDecoder()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps(
        dict(bos_token = "[BOS]", eos_token = "[EOS]", pad_token = "[PAD]", unk_token = "[UNK]")
    ))

    rng = torch.Generator().manual_seed(10)
    weights = {}

    def rand(key, *shape, dtype = torch.bfloat16):
        weights["model.llm." + key] = (torch.randn(shape, generator = rng) * .06).to(dtype)

    def norm(key, *shape):
        weights["model.llm." + key] = torch.ones(shape, dtype = torch.bfloat16)

    rand("embed.weight", vocab, d)
    rand("unembed.weight", vocab, d)
    norm("embed_norm.weight", d)
    norm("norm.weight", d)
    rand("attn_res.output_query", d, dtype = torch.float)
    rand("middle_cycle_cross_pass_queries", 2, d, dtype = torch.float)
    rand("middle_cycle_cross_pass_kv_queries", 2, d, dtype = torch.float)
    for i in range(3):
        p = f"layers.{i}."
        local = i in t["local_layer_ids"]
        if i:
            rand(f"attn_res.attn_queries.{i - 1}", *((d,) if local else (3, d)))
        rand(f"attn_res.mlp_queries.{i}", d)
        for key, ni, no in (
            ("wq_du", d, heads * dh), ("wk_dv", d, kv * dh), ("wv_dv", d, kv * dh),
            ("wr_du", d, heads * d_rel), ("gate_attn", d, heads * dh), ("wo_ud", heads * dh, d),
        ):
            rand(p + f"attn.{key}.weight", no, ni)
        rand(p + "attn.attn_sink_proj.weight", heads, d)
        norm(p + "attn.q_l2_clip.scale", heads, dh)
        norm(p + "attn.rel_l2_clip.scale", heads, d_rel)
        rand(p + "attn.rel_logits_proj.proj", d_rel, extent)
        for which in "kv":
            if local:
                rand(p + f"attn.{which}_sconv.weight", kv * dh, 1, kernel)
            else:
                rand(p + f"attn.{which}_sconv.static_w", kernel, kv * dh)
                rand(p + f"attn.{which}_sconv.weight_proj", kv * kernel, d)
        for n in (["attn"] if local else ["q_input", "k_input", "v_input"]):
            norm(p + f"{n}_norm.weight", d)
        if i == 1:
            for c in ("attn", "k", "v", "mlp"):
                rand(p + f"controllers.{c}.down_weight", rank, 2 * d)
                for kind in ("gain", "forget"):
                    rand(p + f"controllers.{c}.{kind}_weight", d, rank)
        for n in ("attn", "mlp"):
            rand(p + f"{n}_sconv.weight", d, 1, kernel)
        norm(p + "mlp_norm.weight", d)
        rand(p + "mlp.gate.weight", experts, d)
        # Keep routing away from ties in the random tiny model; the unused third
        # expert also exercises the fallback when a down projection has no tokens.
        weights["model.llm." + p + "mlp.gate.bias"] = torch.tensor([2., 0., -2.])
        for n, out, inp in (("w13", 2 * inter, d), ("w2", d, inter)):
            key = "model.llm." + p + f"mlp.experts.{n}_weight"
            if i == 2:
                weights[key] = torch.randn((experts, out, inp), generator = rng).to(torch.float8_e4m3fn)
                weights[key + "_scale"] = torch.full((experts, out, inp // 32), 123, dtype = torch.uint8)
            else:
                weights[key] = torch.randint(256, (experts, out, inp // 2), generator = rng, dtype = torch.uint8)
                weights[key + "_scale"] = torch.full((experts, out, inp // 16), 1., dtype = torch.float8_e4m3fn)
                weights[key + "_amax"] = torch.full((experts,), .02)
                weights[key + "_outer_scale_denominator"] = torch.ones(1)
    save_file(weights, str(path / "model.safetensors"))


@pytest.mark.nogpu
def test_loop_schedule_and_compile_targets(tmp_path):
    write_minimal_checkpoint(tmp_path)
    cfg = Config.from_directory(str(tmp_path), load_method = "python")
    model = Model.from_config(cfg)
    starts = [m for m in model.modules if isinstance(m, Inkling2PassStart)]
    replays = [m for m in model.modules if isinstance(m, Inkling2Replay)]
    assert [m.number for m in starts] == [0, 1, 2]
    assert [m.layer.key for m in replays] == ["model.llm.layers.1"] * 2
    assert all(m.layer is model.find_module("model.llm.layers.1") for m in replays)
    assert all(not m.get_qmaps() for m in replays)
    assert cfg.vocab_size == 125 and cfg.padded_vocab_size == 128

    prelude, bank0, bank1, out = [torch.full((1, 2, 3), i) for i in range(4)]
    params = {"quant_preserve": {"inkling_blocks": (bank0,)}}
    assert starts[0].forward(prelude, params) is prelude
    params["quant_preserve"]["inkling_blocks"] = (bank0, prelude, bank1)
    assert starts[1].forward(out, params) is prelude
    assert params["quant_preserve"]["inkling_previous"] == ((bank1,), out)
    assert params["quant_preserve"]["inkling_blocks"] == (bank0,)
    assert "model.llm.middle_cycle_cross_pass_kv_queries" in model.get_additional_compiled_tensors(cfg)


def test_native_conversion_roundtrip(tmp_path, device):
    from exllamav3.conversion.convert_model import main, parser, prepare

    source, work, out = (tmp_path / name for name in ("source", "work", "exl3"))
    source.mkdir()
    write_minimal_checkpoint(source)

    cal_data = tmp_path / "tokens.safetensors"
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    save_file({"input_ids": torch.tensor([[2, 3, 4, 3, 4, 3, 4, 1]])}, str(cal_data))
    cfg = Config.from_directory(str(source))
    model = Model.from_config(cfg)
    model.load(device = device)
    logits = model.forward(ids, {"last_tokens_only": 1})
    assert logits.shape == (1, 1, cfg.vocab_size) and logits.isfinite().all()
    torch.testing.assert_close(model.forward(ids[:, :-1]), model.forward(ids)[:, :-1], atol = .025, rtol = .025)
    model.unload()

    cli = parser.parse_args([
        "-i", str(source), "-w", str(work), "-o", str(out), "--bits", "4", "--head_bits", "4",
        "--cal_rows", "1", "--cal_cols", "8", "--cal_data", str(cal_data), "--devices", str(torch.device(device).index or 0),
    ])
    args, state, ok, err = prepare(cli)
    assert ok, err
    main(args, state)
    cfg = Config.from_directory(str(out))
    assert cfg.stc.has_tensor("model.llm.layers.1.mlp.experts.0.gate_proj.trellis")
    assert not cfg.stc.has_tensor("model.llm.layers.1.mlp.experts.w13_weight")
    model = Model.from_config(cfg)
    model.load(device = device)
    quant = model.forward(ids, {"last_tokens_only": 1})
    assert quant.shape == logits.shape and quant.isfinite().all()
    assert torch.nn.functional.cosine_similarity(quant.flatten(), logits.flatten(), dim = 0).item() > 0.9
    model.unload()
