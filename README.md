# LockstepODE.jl

[![Build Status](https://github.com/kylebeggs/LockstepODE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kylebeggs/LockstepODE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://kylebeggs.github.io/LockstepODE.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://kylebeggs.github.io/LockstepODE.jl/dev)

LockstepODE.jl is a Julia package for solving multiple ordinary differential equations (ODEs) in lockstep - meaning they share the same timestepping and are synchronized through time. It provides a high-level interface for batching multiple ODE systems into a single solve, with support for threading, different memory layouts, and per-ODE callbacks.

## Features

- **CPU Threading**: Automatic parallelization across ODE systems using `OhMyThreads.jl`
- **Multi-GPU support**: Automatic GPU acceleration for CUDA (NVIDIA), AMDGPU (AMD), Metal (Apple), and oneAPI (Intel) via optional extensions
- **Per-ODE callbacks**: Apply different callbacks to each ODE system or share callbacks across all systems
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

See the [Basic Usage](https://kylebeggs.github.io/LockstepODE.jl/dev/examples/basic_usage/) example for detailed implementation.

### Per-ODE Callbacks

Apply different callbacks to each ODE in your batch. Each ODE can have its own event detection and handling logic while still being solved efficiently in parallel. Supports both `DiscreteCallback` and `ContinuousCallback` from OrdinaryDiffEq.jl.

Example use cases:

- Different reset thresholds per simulation
- Heterogeneous stopping conditions
- Per-instance event logging

See the [Per-ODE Callbacks](https://kylebeggs.github.io/LockstepODE.jl/dev/examples/callbacks/) example for comprehensive examples.

### Memory Layouts

Choose between two memory layouts optimized for different access patterns:

- **PerODE** (default): Each ODE's variables stored contiguously - best for most use cases
- **PerIndex**: Same variable across ODEs stored contiguously - can improve cache locality for certain operations

### Threading Control

Automatic parallelization across ODE systems using `OhMyThreads.jl`. Control threading behavior with `internal_threading` parameter for integration with external parallel workflows or debugging.

### Utility Functions

Built-in functions for common operations:

- `batch_initial_conditions`: Prepare initial conditions for batched solving
- `extract_solutions`: Separate batched solution into individual ODESolution objects
- `create_lockstep_callbacks`: Wrap callbacks for batched execution (usually automatic)

See the [Advanced Configuration](https://kylebeggs.github.io/LockstepODE.jl/dev/examples/advanced_configuration/) example for performance tuning and threading control.

## API Reference

### Main Types

- `LockstepFunction{O,F,B}`: Callable struct that handles the lockstep execution of multiple ODEs
- `PerODE`: Memory layout where each ODE's variables are stored contiguously
- `PerIndex`: Memory layout where variables of the same index across ODEs are stored contiguously

### Constructor

```julia
LockstepFunction(f, ode_size, num_odes; internal_threading=true, ordering=PerODE(), callbacks=nothing)
```

- `f`: ODE function with signature `f(du, u, p, t)`
- `ode_size`: Number of variables per ODE system
- `num_odes`: Number of ODE systems to solve
- `internal_threading`: Whether to use threading for parallel execution (default: `true`)
- `ordering`: Memory layout (`PerODE()` or `PerIndex()`, default: `PerODE()`)
- `callbacks`: Optional callbacks (default: `nothing`). Can be:
  - `nothing`: No callbacks
  - Single callback: Applied to all ODEs
  - Vector of callbacks: Each callback applied to corresponding ODE (length must equal `num_odes`)

### Main Functions

- `batch_initial_conditions(u0, num_odes, ode_size)`: Batch initial conditions for all ODEs
- `extract_solutions(lockstep_func, sol)`: Extract individual ODE solutions from batched solution
- `create_lockstep_callbacks(lockstep_func)`: Create wrapped callbacks (automatically called by `ODEProblem`, rarely needed manually)

### Standard Workflow

1. Create a `LockstepFunction`
2. Use it with standard `OrdinaryDiffEq.jl` functions:

   ```julia
   lockstep_func = LockstepFunction(my_ode!, ode_size, num_odes)
   u0_batched = batch_initial_conditions(u0, num_odes, ode_size)
   prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
   sol = solve(prob, alg)
   individual_sols = extract_solutions(lockstep_func, sol)
   ```

## Performance Tips

1. **Use threading**: Keep `internal_threading=true` (default) for better performance with multiple ODEs
2. **Choose the right layout**: `PerODE` is generally recommended for most use cases
3. **Batch operations**: Use the provided utility functions for efficient batching
4. **Type stability**: Ensure your ODE function is type-stable for best performance

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License.
