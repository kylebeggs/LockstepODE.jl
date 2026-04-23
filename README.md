# LockstepODE.jl

[![Build Status](https://github.com/kylebeggs/LockstepODE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kylebeggs/LockstepODE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://kylebeggs.github.io/LockstepODE.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://kylebeggs.github.io/LockstepODE.jl/dev)

LockstepODE.jl is a Julia package for solving N independent ODEs in parallel with shared timestepping. It provides two execution modes with a unified CommonSolve.jl interface, supporting CPU threading, GPU acceleration, and per-ODE callbacks.

## Features

- **Two execution modes**: Ensemble (N independent integrators) or Batched (single integrator with parallel RHS)
- **CommonSolve.jl interface**: `init`, `solve`, `solve!`, `step!`, `reinit!`
- **CPU threading**: Automatic parallelization via `OhMyThreads.jl`
- **Multi-GPU support**: CUDA, AMDGPU, Metal, oneAPI via optional extensions
- **Per-ODE callbacks**: Different `DiscreteCallback` or `ContinuousCallback` per ODE
- **Flexible memory layouts**: `PerODE` (default) or `PerIndex` for Batched mode
- **ModelingToolkit integration**: Direct support for MTK systems

## Installation

```julia
using Pkg
Pkg.add("LockstepODE")
```

## Quick Start

```julia
using LockstepODE
using OrdinaryDiffEq

# Define a simple harmonic oscillator
function harmonic_oscillator!(du, u, p, t)
    du[1] = u[2]        # dx/dt = v
    du[2] = -u[1]       # dv/dt = -x
end

# Create LockstepFunction for 3 oscillators
lf = LockstepFunction(harmonic_oscillator!, 2, 3)

# Initial conditions for each ODE
u0s = [[1.0, 0.0], [2.0, 0.0], [0.5, 0.5]]

# Create problem and solve (Batched mode by default)
prob = LockstepProblem(lf, u0s, (0.0, 2*pi))
sol = solve(prob, Tsit5())

# Access individual solutions
sol[1]       # First ODE's solution
sol[2](1.5)  # Second ODE interpolated at t=1.5
```

## Execution Modes

### Batched Mode (Default)

Single integrator with batched state vector and parallel RHS evaluation:

```julia
prob = LockstepProblem(lf, u0s, tspan, p)  # Default is Batched
sol = solve(prob, Tsit5())
```

Best for: large N, GPU acceleration, maximum performance.

### Ensemble Mode

N independent ODE integrators with per-ODE adaptive timestepping:

```julia
prob = LockstepProblem{Ensemble}(lf, u0s, tspan, p)
sol = solve(prob, Tsit5())
```

Best for: per-ODE control, complex callbacks, ModelingToolkit integration.

## CommonSolve Interface

```julia
# Initialize without solving
integ = init(prob, Tsit5())

# Manual stepping
step!(integ)
step!(integ, 0.1, true)  # step by dt, stop at t+dt

# Complete the solve
sol = solve!(integ)

# Or solve directly
sol = solve(prob, Tsit5())

# Reinitialize with new conditions
reinit!(integ, new_u0s)
```

## GPU Acceleration

Load your preferred GPU backend and use GPU arrays:

```julia
using LockstepODE
using CUDA  # Or: AMDGPU, Metal, oneAPI

# One CuArray per ODE (ergonomic input)
u0s_gpu = [CuArray([1.0, 0.0]) for _ in 1:1000]

# Use Batched mode for GPU; PerIndex is the GPU-friendly layout
prob = LockstepProblem{Batched}(lf, u0s_gpu, tspan, p; ordering=PerIndex())
sol = solve(prob, Tsit5())  # Runs on GPU
```

The per-ODE `CuArray`s are concatenated into a single contiguous device array
before integration. RHS evaluation is a single KernelAbstractions kernel launch
over the flat state with one thread per ODE — no per-ODE kernel launches or
vector-of-vectors in the hot path.

**Memory ordering matters on GPU.** `PerODE` (the default) stores each ODE's
state contiguously, which means adjacent threads read memory stride-`ode_size`
apart — uncoalesced. `PerIndex` interleaves so that threads at the same
ODE-step read adjacent addresses — coalesced. For small `ode_size` the penalty
is modest; for larger per-ODE state it's the difference between good and bad
bandwidth utilization. Prefer `PerIndex` for GPU.

**Pre-flattened input for large N.** For big batches you can skip the N small
device allocations and pass an already-flat state directly. Its length must be
`num_odes * ode_size` and its byte layout must match the chosen `ordering`:

```julia
u0_flat = CUDA.zeros(Float64, lf.num_odes * lf.ode_size)
# ... fill u0_flat in PerIndex layout ...
prob = LockstepProblem{Batched}(lf, u0_flat, tspan, p; ordering=PerIndex())
```

## Per-ODE Callbacks

Apply different callbacks to each ODE:

```julia
# Different threshold per ODE
callbacks = [
    DiscreteCallback((u,t,integ) -> u[1] > 1.0, integ -> integ.u[1] = 0.0),
    DiscreteCallback((u,t,integ) -> u[1] > 2.0, integ -> integ.u[1] = 0.0),
    DiscreteCallback((u,t,integ) -> u[1] > 3.0, integ -> integ.u[1] = 0.0),
]

lf = LockstepFunction(f!, 2, 3; callbacks=callbacks)
prob = LockstepProblem(lf, u0s, tspan)
```

## Solution Access

```julia
sol = solve(prob, Tsit5())

sol[i]              # i-th ODE solution
sol[i].u            # Time series of states
sol[i].t            # Time points
sol[i](t)           # Interpolate at time t
sol.retcode         # Overall return code

# Extract all states at a specific time
states = extract_at_time(sol, 5.0)
```

## API Reference

### Types

| Type | Description |
|------|-------------|
| `LockstepFunction` | Coordinates N independent ODEs |
| `LockstepProblem{M}` | Problem type (M = `Ensemble` or `Batched`) |
| `LockstepSolution` | Combined solution from N ODEs |
| `Ensemble` | Mode: N independent integrators |
| `Batched` | Mode: Single batched integrator |
| `PerODE` | Memory layout: contiguous per ODE |
| `PerIndex` | Memory layout: strided same-index |
| `BatchedOpts` | Options for Batched mode |

### Constructor Signatures

```julia
# LockstepFunction
LockstepFunction(f, ode_size, num_odes; callbacks=nothing)

# LockstepProblem (Batched mode - default)
LockstepProblem(lf, u0s, tspan, p=nothing)

# LockstepProblem (Batched mode)
LockstepProblem{Batched}(lf, u0s, tspan, p=nothing; ordering=PerODE(), internal_threading=true)
```

### Functions

| Function | Description |
|----------|-------------|
| `init(prob, alg; kwargs...)` | Create integrator |
| `solve(prob, alg; kwargs...)` | Solve to completion |
| `solve!(integ)` | Complete initialized integrator |
| `step!(integ)` | Advance one step |
| `step!(integ, dt, stop_at_tdt)` | Advance by dt |
| `reinit!(integ, u0s; kwargs...)` | Reinitialize integrator |
| `extract_at_time(sol, t)` | Extract all states at time t |

## Performance Tips

1. **Choose the right mode**: Ensemble for control, Batched for performance
2. **Use threading**: Keep `internal_threading=true` (default)
3. **GPU for large N**: Use Batched mode with GPU arrays for N > 100. Pass `ordering=PerIndex()` for coalesced memory access; pass a pre-flattened `CuArray` of length `num_odes * ode_size` to avoid N small device allocations.
4. **Type stability**: Ensure your ODE function is type-stable

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License.
