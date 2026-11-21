# Inkling 2 Mini (experimental)

This fork adds text-only support for `thinkingmachines/Inkling-2-Mini`, architecture
`Inkling2LoopForConditionalGeneration`. The fixed three-pass core reuses its
physical weights and takes messages from the preceding pass. The decoder supports
both the original packed NVFP4/MXFP8 experts and native EXL3 tensors.

## Make an EXL3 quant

Download the official checkpoint and its tokenizer/config files to a local
directory. Then run the regular converter from this branch:

```sh
python convert.py \
  --in_dir /path/to/Inkling-2-Mini \
  --work_dir /path/to/new-inkling-conversion-work \
  --out_dir /path/to/new-inkling-exl3 \
  --bits 3.5 --head_bits 6 --devices 0 \
  --cal_rows 32 --cal_cols 512
```

The bitrate and calibration size above are starting points, not quality
recommendations. Use separate, fresh working and output directories. The original
expert matrices are dequantized one physical layer at a time. The converter fits
the core's weights on their first visit, then replays those same quantized
weights to calibrate the coda and output head on the final pass's state. It does
not pool Hessians from all three visits. The router, convolutions, and loop
controllers are retained as-is. The vision tower and DSPark/MTP weights are
not included in the converted checkpoint.

## Try text inference

```python
import torch

from exllamav3 import Config, Model, Tokenizer

config = Config.from_directory("/path/to/new-inkling-exl3")
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
model.load(device="cuda:0")

# Raw continuation for a smoke test (not chat formatting).
ids = tokenizer.encode("The capital of France is", add_bos=False)
start = ids.shape[1]
for _ in range(8):
    logits = model.forward(ids, {"last_tokens_only": 1})
    next_id = logits[:, -1].argmax(dim=-1)
    if next_id.item() in config.eos_token_id_list:
        break
    ids = torch.cat((ids, next_id.cpu()[:, None]), dim=1)
print(tokenizer.decode(ids[0, start:]))
```

This is a reference backend. It recomputes the prompt for every new token;
paged KV caching, the standard `Generator`/TabbyAPI, vision, speculative
decoding, and tensor parallelism are not supported. Use short contexts for
initial tests. The nominal 1M context limit in the checkpoint is not a
performance or memory claim for this implementation.

For architecture and checkpoint details, see the [official Transformers
implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2_loop)
and the [llama.cpp Mini port](https://github.com/ggml-org/llama.cpp/commit/29b52d398b294509746ec0a3f28948a3107c0213).
