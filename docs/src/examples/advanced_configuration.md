# Advanced Configuration

This page covers performance optimization and advanced configuration options in LockstepODE.jl v2.0, including memory layout selection, threading control, and mode selection.

## Execution Modes

LockstepODE v2.0 provides two execution modes with different performance characteristics:

### Batched Mode (Default)

Single integrator with batched state vector and parallel RHS evaluation:

```julia
prob = LockstepProblem(lf, u0s, tspan, ps)  # Default is Batched
# or explicitly with options:
prob = LockstepProblem{Batched}(lf, u0s, tspan, ps;
    ordering = PerODE(),        # Memory layout
    internal_threading = true   # CPU threading
)
```

**Characteristics:**
- Single timestepping for all ODEs
- Parallel RHS evaluation
- GPU acceleration support
- Maximum throughput

**When to use:**
- Large N (100+ ODEs)
- GPU acceleration desired
- All ODEs have similar dynamics
- Per-ODE control not needed during solve

### Ensemble Mode

N independent ODE integrators, each with adaptive timestepping:

```julia
prob = LockstepProblem{Ensemble}(lf, u0s, tspan, ps)
```

**Characteristics:**
- Per-ODE adaptive timestepping
- Full integrator access per ODE
- Standard OrdinaryDiffEq callbacks
- Best for ModelingToolkit integration

**When to use:**
- Per-ODE introspection needed during integration
- Complex per-ODE callbacks
- ODEs with very different dynamics/stiffness
- Moderate N (< 100 ODEs)

---

## Memory Layouts (Batched Mode)

Batched mode supports two memory layouts for organizing batched ODE data.

### Memory Layout Comparison

For a system with 3 ODEs, each with 2 variables (u1, u2):

**PerODE (default)**: Variables for each ODE stored contiguously
```
[ODE1_u1, ODE1_u2, ODE2_u1, ODE2_u2, ODE3_u1, ODE3_u2]
```

**PerIndex**: Same variable index across ODEs stored contiguously
```
[ODE1_u1, ODE2_u1, ODE3_u1, ODE1_u2, ODE2_u2, ODE3_u2]
```

### When to Use Each Layout

**PerODE (recommended for most cases)**:
- Better cache locality when computing derivatives for a single ODE
- More intuitive data organization
- Default behavior

**PerIndex**:
- Better cache locality for operations across all ODEs on the same variable
- Useful when vectorizing operations across ODEs
- Potentially better SIMD opportunities

### Using Different Layouts

```julia
using LockstepODE
using OrdinaryDiffEq

function my_ode!(du, u, p, t)
    du[1] = p * u[2]
    du[2] = -u[1]
end

lf = LockstepFunction(my_ode!, 2, 4)

u0s = [[1.0, 0.0], [2.0, 1.0], [0.5, -0.5], [1.5, 0.8]]
ps = [1.0, 1.2, 0.8, 1.1]

# PerODE layout (default)
prob_per_ode = LockstepProblem{Batched}(lf, u0s, (0.0, 10.0), ps;
    ordering = PerODE()
)

# PerIndex layout
prob_per_index = LockstepProblem{Batched}(lf, u0s, (0.0, 10.0), ps;
    ordering = PerIndex()
)

sol = solve(prob_per_ode, Tsit5())
```

### Performance Considerations

```julia
using BenchmarkTools

function benchmark_layouts(lf, u0s, tspan, ps)
    # PerODE layout
    prob_per_ode = LockstepProblem{Batched}(lf, u0s, tspan, ps; ordering=PerODE())

    # PerIndex layout
    prob_per_index = LockstepProblem{Batched}(lf, u0s, tspan, ps; ordering=PerIndex())

    println("PerODE layout:")
    @btime solve($prob_per_ode, Tsit5())

    println("\nPerIndex layout:")
    @btime solve($prob_per_index, Tsit5())
end
```

**Typical observations**:
- For small ODE systems (ode_size <= 10): Both layouts perform similarly
- For large ODE systems (ode_size > 100): PerODE usually faster
- For many small ODEs with simple dynamics: PerIndex may be faster

---

## Threading Control

### Ensemble Mode Threading

Ensemble mode uses `Threads.@threads` to parallelize integrator creation and stepping:

```julia
# Threading is automatic based on Julia thread count
prob = LockstepProblem(lf, u0s, tspan, ps)  # Uses available threads
```

Control with Julia's thread count:
```bash
JULIA_NUM_THREADS=8 julia
```

### Batched Mode Threading

Batched mode uses `OhMyThreads.jl` for parallel RHS evaluation:

```julia
# Enable threading (default)
prob = LockstepProblem{Batched}(lf, u0s, tspan, ps; internal_threading=true)

# Disable threading
prob = LockstepProblem{Batched}(lf, u0s, tspan, ps; internal_threading=false)
```

### When to Disable Threading

Disable internal threading when:
- You're already parallelizing at a higher level
- Running in a single-threaded environment
- Want deterministic behavior for debugging

```julia
using Distributed
@everywhere using LockstepODE, OrdinaryDiffEq

# External parallelization with internal threading disabled
@everywhere function solve_variant(param)
    lf = LockstepFunction(my_ode!, 2, 10)
    u0s = [[1.0, 0.0] for _ in 1:10]
    prob = LockstepProblem{Batched}(lf, u0s, (0.0, 10.0), param;
        internal_threading = false  # Avoid over-subscription
    )
    return solve(prob, Tsit5())
end

params = [0.8, 0.9, 1.0, 1.1, 1.2]
results = pmap(solve_variant, params)
```

---

## GPU Acceleration

GPU acceleration is available in Batched mode:

```julia
using LockstepODE
using CUDA  # Or: AMDGPU, Metal, oneAPI

function my_ode!(du, u, p, t)
    du[1] = p * u[2]
    du[2] = -u[1]
end

lf = LockstepFunction(my_ode!, 2, 1000)

# Create GPU initial conditions
u0s_gpu = [CuArray([1.0, 0.0]) for _ in 1:1000]
ps_gpu = CuArray(ones(1000))

# Use Batched mode (required for GPU)
prob = LockstepProblem{Batched}(lf, u0s_gpu, (0.0, 10.0), ps_gpu)
sol = solve(prob, Tsit5())
```

### Supported Backends

| Package | GPU Type | Array Type |
|---------|----------|------------|
| CUDA.jl | NVIDIA | `CuArray` |
| AMDGPU.jl | AMD | `ROCArray` |
| Metal.jl | Apple Silicon | `MtlArray` |
| oneAPI.jl | Intel | `oneArray` |

Backend selection is automatic based on array type.

---

## Performance Tuning

### Choosing a Mode

```julia
# Small N, per-ODE control needed: Ensemble
if num_odes < 50 || need_per_ode_callbacks
    prob = LockstepProblem(lf, u0s, tspan, ps)

# Large N, GPU, or maximum throughput: Batched
else
    prob = LockstepProblem{Batched}(lf, u0s, tspan, ps)
end
```

### Benchmarking Configurations

```julia
using BenchmarkTools

function benchmark_modes(lf, u0s, tspan, ps)
    # Ensemble mode
    prob_ensemble = LockstepProblem(lf, u0s, tspan, ps)

    # Batched modes
    prob_batched_perode = LockstepProblem{Batched}(lf, u0s, tspan, ps;
        ordering=PerODE(), internal_threading=true)
    prob_batched_perindex = LockstepProblem{Batched}(lf, u0s, tspan, ps;
        ordering=PerIndex(), internal_threading=true)

    println("Ensemble mode:")
    @btime solve($prob_ensemble, Tsit5())

    println("\nBatched (PerODE + Threading):")
    @btime solve($prob_batched_perode, Tsit5())

    println("\nBatched (PerIndex + Threading):")
    @btime solve($prob_batched_perindex, Tsit5())
end
```

### General Performance Tips

1. **Mode selection**: Ensemble for control, Batched for throughput
2. **Threading**: Keep enabled unless nested parallelism
3. **Layout**: Start with PerODE, benchmark PerIndex for specific workloads
4. **GPU**: Use Batched mode with GPU arrays for N > 100
5. **Type stability**: Ensure ODE function is type-stable (`@code_warntype`)

---

## Thread Safety

When using threading (Ensemble mode or Batched with `internal_threading=true`), ensure your ODE function is thread-safe:

```julia
# Thread-safe: Only modifies local arrays (du, u)
function safe_ode!(du, u, p, t)
    du[1] = p * u[2]
    du[2] = -u[1]
end

# Potentially unsafe: Modifies shared state
global_counter = Ref(0)
function unsafe_ode!(du, u, p, t)
    global_counter[] += 1  # Race condition!
    du[1] = p * u[2]
    du[2] = -u[1]
end
```

If you must modify shared state, use thread-safe primitives:

```julia
using Base.Threads: Atomic, atomic_add!

counter = Atomic{Int}(0)
function thread_safe_counting_ode!(du, u, p, t)
    atomic_add!(counter, 1)  # Thread-safe increment
    du[1] = p * u[2]
    du[2] = -u[1]
end
```

---

## Summary

### Configuration Options

| Mode | Option | Values | Default | Description |
|------|--------|--------|---------|-------------|
| Batched | `ordering` | `PerODE()`, `PerIndex()` | `PerODE()` | Memory layout |
| Batched | `internal_threading` | `true`, `false` | `true` | CPU threading for RHS |

### Mode Comparison

| Feature | Ensemble | Batched |
|---------|----------|---------|
| Integrators | N independent | 1 batched |
| Timestepping | Per-ODE adaptive | Shared |
| GPU support | No | Yes |
| Per-ODE control | Full | Limited |
| Best for N | < 100 | > 100 |

### Recommended Workflow

1. Start with Batched mode (default) - best for most use cases
2. If you need per-ODE adaptive timesteps or complex callbacks, try Ensemble mode
3. For GPU, use Batched mode with GPU arrays
4. Benchmark different configurations for your specific workload

For basic usage patterns, see [Basic Usage](basic_usage.md). For event handling, see [Callbacks](callbacks.md).
