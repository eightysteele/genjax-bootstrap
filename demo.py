import timeit

import jax
import jax.numpy as jnp
import genjax
from genjax import beta, flip, gen, Target, ChoiceMap
from genjax.inference.smc import ImportanceK

@gen
def beta_bernoulli(α, β):
    """Define the generative model."""
    p = beta(α, β) @ "p"
    v = flip(p) @ "v"
    return v

def run_inference(obs: bool, platform='cpu'):
    """Estimate `p` over 50 independent trials of SIR (K = 50 particles)."""
    # Set the device
    device = jax.devices(platform)[0]
    key = jax.random.PRNGKey(314159)
    key = jax.device_put(key, device)

    # JIT compilation will be target-specific based on device (CPU or GPU)
    @jax.jit
    def execute_inference():
        # Inference query with the a model, arguments, and constraints
        posterior = Target(beta_bernoulli, (2.0, 2.0), ChoiceMap.d({"v": obs}))

        # Use a library algorithm, or design your own—more on that in the docs!
        alg = ImportanceK(posterior, k_particles=50)

        # Everything is JAX compatible by default—jit, vmap, etc.
        skeys = jax.random.split(key, 50)
        _, p_chm = jax.vmap(
            alg.random_weighted, in_axes=(0, None))(skeys, posterior)

        return jnp.mean(p_chm["p"])

    return execute_inference


n = 1000
print(f"\nStarting GenJax demo with {n} runs...")

# CPU compile, execute, benchmark, profile
cpu_jit = jax.jit(run_inference(True, 'cpu'))
cpu_jit().block_until_ready()
ms = timeit.timeit(
    'cpu_jit()',
    globals=globals(),
    number=n
) / n * 1000
print(f"CPU: Average runtime over {n} runs = {ms} (ms)")

# GPU compile, execute, benchmark, profile
gpu_jit = jax.jit(run_inference(True, 'cpu'))
gpu_jit().block_until_ready()
try:
    jax.devices('gpu')
    ms = timeit.timeit(
        'gpu_jit',
        globals=globals(),
        number=n
    ) / n * 1000
    print(f"GPU: Average runtime over {n} runs = {ms} (ms)")
except RuntimeError as e:
    print("(No GPU device on host)")

print("Done!")
