"""Compare HLO for naive jnp.dot(fp8, fp8) vs jax.nn.scaled_matmul.

Must run on GPU — scaled_matmul has no CPU lowering rule.
Run with: modal run check_fp8_hlo.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cuda13]==0.9.2", "numpy==2.4.5")
)

app = modal.App("check-fp8-hlo")


@app.function(image=image, gpu="B200", timeout=5 * 60)
def check_hlo():
    import jax
    import jax.numpy as jnp

    M, K, N = 16384, 2304, 2304

    a_fp8 = jnp.ones((M, K), dtype=jnp.float8_e4m3fn)
    b_fp8 = jnp.ones((K, N), dtype=jnp.float8_e4m3fn)

    a_scales = jnp.ones((1, M, K // 128), dtype=jnp.float32)   # (1, 16384, 18)
    b_scales = jnp.ones((1, N, K // 128), dtype=jnp.float32)   # (1, 2304, 18)

    def naive_dot(a, b):
        return jnp.dot(a, b, preferred_element_type=jnp.float32)

    def scaled_mm(a, b, a_s, b_s):
        return jax.nn.scaled_matmul(a[None], b.T[None], a_s, b_s,
                                    preferred_element_type=jnp.float32)[0]

    hlo_naive          = jax.jit(naive_dot).lower(a_fp8, b_fp8).as_text()
    hlo_naive_compiled = jax.jit(naive_dot).lower(a_fp8, b_fp8).compile().as_text()

    def print_relevant(label, hlo):
        print(f"\n{'='*60}")
        print(f"  {label}")
        print('='*60)
        for line in hlo.splitlines():
            if any(k in line for k in ("dot_general", "convert", "algorithm", "f8e4m3", "float8", "custom_call")):
                print(line)

    print_relevant("naive jnp.dot — uncompiled StableHLO", hlo_naive)
    print_relevant("naive jnp.dot — compiled HLO (after XLA passes)", hlo_naive_compiled)

    print("\n\n--- full compiled HLO: naive dot ---")
    print(hlo_naive_compiled)


@app.local_entrypoint()
def main():
    check_hlo.remote()
