```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: LockstepODE.jl Docs
  tagline: Stepping Through ODEs in Lockstep
---
```

## Overview

LockstepODE.jl is a Julia package for solving multiple ordinary differential equations (ODEs) in lockstep - meaning they share the same timestepping and are synchronized through time. It provides a high-level interface for batching multiple ODE systems into a single solve, with support for CPU threading and GPU acceleration, different memory layouts, and per-ODE callbacks.

## Features

- **CPU Threading**: Automatic CPU parallelization across ODE systems using `OhMyThreads.jl`
- **GPU Acceleration**: Automatic GPU support via KernelAbstractions.jl for CUDA (NVIDIA), AMDGPU (AMD), Metal (Apple), and oneAPI (Intel)
- **Callbacks**: Apply different callbacks to each ODE system or share callbacks across all systems
- **Flexible memory layouts**: Support for both per-ODE (`PerODE`) and per-index (`PerIndex`) data organization
- **Standard workflow**: Uses standard `OrdinaryDiffEq.jl` workflow with `ODEProblem` and `solve`
- **Utility functions**: Built-in functions for batching initial conditions and extracting solutions

## Installation

```julia
using Pkg
Pkg.add("LockstepODE")
```

Or from the Julia REPL:

```julia
] add LockstepODE
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

# Create a lockstep function for 3 oscillators
num_odes = 3
u0_single = [1.0, 0.0]  # Single initial condition
u0_batched = repeat(u0_single, num_odes)  # Batch for all oscillators
lockstep_func = LockstepFunction(harmonic_oscillator!, length(u0_single), num_odes)

# Use standard OrdinaryDiffEq.jl workflow
prob = ODEProblem(lockstep_func, u0_batched, (0.0, 2π))
sol = solve(prob, Tsit5())

# Extract individual solutions
individual_sols = extract_solutions(lockstep_func, sol)
```

## GPU Acceleration

LockstepODE.jl automatically detects and uses GPU arrays when available. Simply load your preferred GPU backend and convert your arrays:

```julia
using LockstepODE
using CUDA  # Or: AMDGPU, Metal, oneAPI

# Everything else stays the same - just use GPU arrays
u0_batched = CuArray(u0_batched)  # Move to GPU
prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
sol = solve(prob, Tsit5())  # Automatically runs on GPU
```

Supported backends:
- **CUDA.jl**: NVIDIA GPUs
- **AMDGPU.jl**: AMD GPUs (ROCm)
- **Metal.jl**: Apple Silicon GPUs
- **oneAPI.jl**: Intel GPUs

All backends are optional - only install what you need. No code changes required; backend selection is automatic based on array type.

## Key Features Explained

### Multiple Initial Conditions

Solve the same ODE system with different starting points simultaneously. LockstepODE batches your initial conditions and parameters into a single efficient solve, then extracts individual solutions. Perfect for parameter sweeps, ensemble simulations, or Monte Carlo methods.

See the [Basic Usage](examples/basic_usage.md) example for detailed implementation.

### Callbacks

Apply different callbacks to each ODE in your batch. Each ODE can have its own event detection and handling logic while still being solved efficiently in parallel. Supports both `DiscreteCallback` and `ContinuousCallback` from OrdinaryDiffEq.jl.

Example use cases:
- Different reset thresholds per simulation
- Heterogeneous stopping conditions
- Per-instance event logging

See the [Callbacks](examples/callbacks.md) example for comprehensive examples.

### Memory Layouts

Choose between two memory layouts optimized for different access patterns:
- **PerODE** (default): Each ODE's variables stored contiguously - best for most use cases
- **PerIndex**: Same variable across ODEs stored contiguously - can improve cache locality for certain operations

### Threading Control

CPU parallelization across ODE systems using `OhMyThreads.jl`, with automatic GPU acceleration when GPU arrays are used. Control threading behavior with `internal_threading` parameter for integration with external parallel workflows or debugging.

### Utility Functions

Built-in functions for common operations:
- `batch_initial_conditions`: Prepare initial conditions for batched solving
- `extract_solutions`: Separate batched solution into individual ODESolution objects
- `create_lockstep_callbacks`: Wrap callbacks for batched execution (usually automatic)

See the [Advanced Configuration](examples/advanced_configuration.md) example for performance tuning and threading control.