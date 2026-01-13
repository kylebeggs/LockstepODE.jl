```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: LockstepODE.jl Docs
  tagline: Solving N Independent ODEs in Parallel
---
```

## Overview

LockstepODE.jl is a Julia package for solving N independent ODEs in parallel with shared timestepping. It provides two execution modes with a unified CommonSolve.jl interface, supporting CPU threading, GPU acceleration, and per-ODE callbacks.

## Features

- **Two execution modes**: Ensemble (N independent integrators) or Batched (single integrator with parallel RHS)
- **CommonSolve.jl interface**: `init`, `solve`, `solve!`, `step!`, `reinit!`
- **CPU threading**: Automatic parallelization via `OhMyThreads.jl`
- **GPU acceleration**: CUDA, AMDGPU, Metal, oneAPI via optional extensions
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

## GPU Acceleration

Use Batched mode with GPU arrays:

```julia
using LockstepODE
using CUDA  # Or: AMDGPU, Metal, oneAPI

u0s_gpu = [CuArray([1.0, 0.0]) for _ in 1:1000]
prob = LockstepProblem{Batched}(lf, u0s_gpu, tspan, p)
sol = solve(prob, Tsit5())  # Runs on GPU
```

Supported backends:
- **CUDA.jl**: NVIDIA GPUs
- **AMDGPU.jl**: AMD GPUs (ROCm)
- **Metal.jl**: Apple Silicon GPUs
- **oneAPI.jl**: Intel GPUs

## Key Features

### CommonSolve Interface

```julia
integ = init(prob, Tsit5())
step!(integ)
sol = solve!(integ)
reinit!(integ, new_u0s)
```

### Per-ODE Callbacks

```julia
callbacks = [
    DiscreteCallback((u,t,integ) -> u[1] > 1.0, integ -> integ.u[1] = 0.0),
    DiscreteCallback((u,t,integ) -> u[1] > 2.0, integ -> integ.u[1] = 0.0),
]
lf = LockstepFunction(f!, 2, 2; callbacks=callbacks)
```

### Memory Layouts (Batched Mode)

- **PerODE** (default): Each ODE's variables stored contiguously
- **PerIndex**: Same variable across ODEs stored contiguously

### Solution Access

```julia
sol[i]              # i-th ODE solution
sol[i].u            # Time series
sol[i](t)           # Interpolation
extract_at_time(sol, t)  # All states at time t
```

## Documentation

- [Getting Started](getting_started.md): Complete tutorial
- [API Reference](api.md): Full API documentation
- [Examples](examples/basic_usage.md): Usage examples

## Contents

```@contents
Pages = ["getting_started.md", "api.md", "examples/basic_usage.md", "examples/callbacks.md", "examples/advanced_configuration.md"]
Depth = 2
```
