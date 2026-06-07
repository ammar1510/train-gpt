"""Modal B200 wrapper for sampling from a checkpoint on the volume.

Reads a checkpoint directly off the `train-gpt-checkpoints` volume and runs
generation on a B200, so you don't have to download the ~3.5 GB FULL-config
pickle just to see a few samples. The actual generation logic lives in
generate.py (shared with the local CLI) — this only handles GPU provisioning,
volume mounting, and tokenization.

Usage:
    modal run generate_modal.py --checkpoint step_10000.pkl \
        --prompt "The mitochondria is" --max-new-tokens 100

    modal run generate_modal.py --checkpoint step_10000.pkl --prompt "Once upon" \
        --temperature 0                      # greedy
"""
import modal

CKPT_VOLUME_NAME = "train-gpt-checkpoints"
REMOTE_CKPT_DIR = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda13]==0.9.2",
        "numpy==2.4.5",
        "tiktoken==0.13.0",
    )
    .add_local_python_source("config", "model", "kernels", "generate", "sft_data")
)

app = modal.App("train-gpt-generate")
ckpt_vol = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    timeout=15 * 60,
    volumes={REMOTE_CKPT_DIR: ckpt_vol},
)
def run_generate(
    checkpoint: str,
    prompt: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    num_samples: int = 1,
    seed: int = 0,
    prepend_eot: bool = True,
    chat: bool = False,
) -> list[str]:
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

    from pathlib import Path

    import jax
    import tiktoken

    from generate import generate, load_checkpoint
    from sft_data import format_chat_prompt

    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if chat and not prompt:
        raise ValueError("--chat requires a non-empty --prompt")

    params, cfg, step = load_checkpoint(Path(REMOTE_CKPT_DIR) / checkpoint)
    enc = tiktoken.get_encoding("gpt2")  # matches prepare_data.py
    eot = enc.eot_token

    print(f"loaded {checkpoint} (step {step}) on {jax.default_backend()}; "
          f"params={sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
    if cfg.vocab_size != enc.n_vocab:
        print(f"  WARNING: cfg.vocab_size={cfg.vocab_size} != tokenizer "
              f"n_vocab={enc.n_vocab}; output may be garbage")

    if chat:
        # Chat template (matches sft_data.build_example's first turn).
        prompt_ids = format_chat_prompt(prompt, enc, eot)
    else:
        prompt_ids = enc.encode_ordinary(prompt) if prompt else []
        if prepend_eot:
            prompt_ids = [eot] + prompt_ids
        if not prompt_ids:
            prompt_ids = [eot]

    outputs: list[str] = []
    for i in range(num_samples):
        new_ids = generate(
            params, cfg, prompt_ids, max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            seed=seed + i, eot_token=eot, stop_on_eot=True,
        )
        if chat:
            outputs.append(f"User: {prompt}\nAssistant: {enc.decode(new_ids).strip()}")
        else:
            outputs.append(prompt + enc.decode(new_ids))
    return outputs


@app.local_entrypoint()
def main(
    checkpoint: str,
    prompt: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    num_samples: int = 1,
    seed: int = 0,
    prepend_eot: bool = True,
    chat: bool = False,
):
    samples = run_generate.remote(
        checkpoint=checkpoint,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_samples=num_samples,
        seed=seed,
        prepend_eot=prepend_eot,
        chat=chat,
    )
    for i, text in enumerate(samples):
        header = f"sample {i + 1}/{len(samples)}" if len(samples) > 1 else "sample"
        print(f"\n{'─' * 60}\n{header}\n{'─' * 60}")
        print(text)
