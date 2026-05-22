"""Download FineWeb-Edu (10BT slice), tokenize with GPT-2 BPE, write uint16 .bin.

Runs on a Modal container; downloads the resulting .bin files back to ./data.

Outputs (local):
    data/fineweb-edu-10BT-train.bin
    data/fineweb-edu-10BT-val.bin

Re-running is idempotent: dataset + HF cache are persisted on a Modal Volume,
and existing .bin files of the expected size are not rewritten.

One-time setup (HF auth — needed to avoid rate limits on the Hub):
    modal secret create huggingface HF_TOKEN=hf_xxx

Usage:
    modal run prepare_data.py
    modal run prepare_data.py --num-proc 16 --val-frac 0.005
"""
import os
from pathlib import Path

import modal

DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"
ENCODING = "gpt2"

REMOTE_DATA_DIR = "/data"
REMOTE_HF_CACHE = "/hf-cache"
LOCAL_OUT_DIR = Path("data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.5",
        "tiktoken==0.13.0",
        "datasets==4.8.5",
    )
)

app = modal.App("prepare-fineweb-edu")

data_vol = modal.Volume.from_name("fineweb-edu-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    cpu=16.0,
    memory=32 * 1024,
    volumes={REMOTE_DATA_DIR: data_vol, REMOTE_HF_CACHE: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    timeout=6 * 60 * 60,
)
def prepare(
    num_proc: int = 16,
    val_frac: float = 0.005,
    shard_tokens: int = 10**8,
    seed: int = 0,
) -> dict[str, int]:
    import numpy as np
    import tiktoken
    from datasets import load_dataset

    os.environ["HF_HOME"] = REMOTE_HF_CACHE
    os.environ["HF_DATASETS_CACHE"] = f"{REMOTE_HF_CACHE}/datasets"

    enc = tiktoken.get_encoding(ENCODING)
    eot = enc.eot_token
    assert enc.n_vocab <= 2**16, "uint16 requires vocab <= 65536"

    print(f"loading dataset {DATASET}:{SUBSET} ...")
    ds = load_dataset(DATASET, name=SUBSET, split="train")
    split = ds.train_test_split(test_size=val_frac, seed=seed, shuffle=True)
    splits = {"train": split["train"], "val": split["test"]}

    def tokenize(doc):
        ids = [eot]
        ids.extend(enc.encode_ordinary(doc["text"]))
        return {"ids": ids, "len": len(ids)}

    out_dir = Path(REMOTE_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}

    for name, dset in splits.items():
        out_path = out_dir / f"fineweb-edu-10BT-{name}.bin"
        print(f"\ntokenizing {name}: {len(dset):,} docs -> {out_path}")

        tokenized = dset.map(
            tokenize,
            remove_columns=["text"],
            num_proc=num_proc,
            desc=f"tokenize {name}",
        )
        total = int(np.sum(tokenized["len"], dtype=np.int64))
        print(f"  total tokens: {total:,}")
        sizes[name] = total

        if out_path.exists() and out_path.stat().st_size == total * 2:
            print("  already up-to-date, skipping write")
            continue

        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total,))
        idx = 0
        n_shards = max(1, total // shard_tokens)
        for shard_i in range(n_shards):
            batch = tokenized.shard(
                num_shards=n_shards, index=shard_i, contiguous=True
            ).with_format("numpy")
            ids = np.concatenate(batch["ids"]).astype(np.uint16)
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
            print(
                f"  shard {shard_i + 1}/{n_shards}  {idx:,}/{total:,} tokens "
                f"({100 * idx / total:.1f}%)"
            )
        arr.flush()
        assert idx == total, f"wrote {idx} but expected {total}"

    data_vol.commit()
    print("\nremote prep done.")
    return sizes


@app.local_entrypoint()
def main(
    num_proc: int = 16,
    val_frac: float = 0.005,
    shard_tokens: int = 10**8,
    seed: int = 0,
):
    sizes = prepare.remote(
        num_proc=num_proc,
        val_frac=val_frac,
        shard_tokens=shard_tokens,
        seed=seed,
    )

    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, total in sizes.items():
        fname = f"fineweb-edu-10BT-{name}.bin"
        local_path = LOCAL_OUT_DIR / fname
        expected_bytes = total * 2

        if local_path.exists() and local_path.stat().st_size == expected_bytes:
            print(f"{fname}: local copy already up-to-date ({expected_bytes:,} bytes)")
            continue

        print(f"downloading {fname} ({expected_bytes:,} bytes) ...")
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        with open(tmp_path, "wb") as f:
            for chunk in data_vol.read_file(fname):
                f.write(chunk)
        tmp_path.replace(local_path)

        got = local_path.stat().st_size
        assert got == expected_bytes, f"{fname}: got {got} bytes, expected {expected_bytes}"
        print(f"  {fname}: {got:,} bytes ok")

    print("\ndone.")
