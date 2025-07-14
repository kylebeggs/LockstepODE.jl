# LockstepODE.jl

[![Build Status](https://github.com/Kyle Beggs/LockstepODE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Kyle Beggs/LockstepODE.jl/actions/workflows/CI.yml?query=branch%3Amain)

LockstepODE.jl is a Julia package for efficiently solving multiple coupled ordinary differential equations (ODEs) in lockstep. It provides a high-level interface for batching multiple ODE systems and solving them simultaneously, with support for both threading and different memory layouts.

## Features

- **Efficient batching**: Solve multiple ODE systems simultaneously
- **Threading support**: Automatic parallelization across ODE systems using `OhMyThreads.jl`
- **Flexible memory layouts**: Support for both per-ODE (`PerODE`) and per-index (`PerIndex`) data organization
- **Standard workflow**: Uses standard `OrdinaryDiffEq.jl` workflow with `ODEProblem` and `solve`
- **Easy integration**: Works seamlessly with `OrdinaryDiffEq.jl` and the SciML ecosystem
- **Utility functions**: Built-in functions for batching initial conditions, parameters, and extracting solutions

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
lockstep_func = LockstepFunction(harmonic_oscillator!, u0_batched, num_odes)

# Use standard OrdinaryDiffEq.jl workflow
prob = ODEProblem(lockstep_func, u0_batched, (0.0, 2π))
sol = solve(prob, Tsit5())

# Extract individual solutions
individual_sols = extract_solutions(lockstep_func, sol)
```

## Usage Examples

### Basic Usage with Multiple Initial Conditions

```julia
using LockstepODE
using OrdinaryDiffEq

# Define your ODE function
function my_ode!(du, u, p, t)
    du[1] = p * u[2]
    du[2] = -u[1]
end

# Set up multiple initial conditions
num_odes = 4
u0_vec = [[1.0, 0.0], [2.0, 1.0], [0.5, -0.5], [1.5, 0.8]]
u0_batched = vcat(u0_vec...)  # Flatten to single vector
p_batched = [1.0, 1.2, 0.8, 1.1]  # Different parameters for each ODE

# Create the lockstep function
lockstep_func = LockstepFunction(my_ode!, u0_batched, num_odes)

# Use standard OrdinaryDiffEq.jl workflow
prob = ODEProblem(lockstep_func, u0_batched, (0.0, 10.0), p_batched)
sol = solve(prob, Tsit5())

# Extract and analyze individual solutions
individual_sols = extract_solutions(lockstep_func, sol)
for (i, isol) in enumerate(individual_sols)
    println("ODE $i final state: ", isol.u[end])
end
```

### Using Different Memory Layouts

```julia
# Per-index layout (can be more cache-friendly for some operations)
lockstep_func_per_index = LockstepFunction(
    my_ode!, 
    u0_batched,
    num_odes;
    ordering=PerIndex()
)
prob_per_index = ODEProblem(lockstep_func_per_index, u0_batched, (0.0, 10.0), p_batched)
```

### Threading Control

```julia
# Disable internal threading (useful if you want to control threading externally)
lockstep_func_no_threading = LockstepFunction(
    my_ode!, 
    u0_batched,
    num_odes;
    internal_threading=false
)
prob_no_threading = ODEProblem(lockstep_func_no_threading, u0_batched, (0.0, 10.0), p_batched)
```

### Using Utility Functions

```julia
# Create a lockstep function first
lockstep_func = LockstepFunction(my_ode!, zeros(8), 4)  # 4 ODEs, 2 variables each

# Batch initial conditions from individual vectors
u0_vec = [[1.0, 0.0], [2.0, 1.0], [0.5, -0.5], [1.5, 0.8]]
u0_batched = batch_initial_conditions(lockstep_func, u0_vec)

# Or repeat a single initial condition
u0_single = [1.0, 0.0]
u0_repeated = batch_initial_conditions(lockstep_func, u0_single)

# Batch parameters
p_vec = [1.0, 1.2, 0.8, 1.1]
p_batched = batch_parameters(lockstep_func, p_vec)
```

## API Reference

### Main Types

- `LockstepFunction{O,F}`: Callable struct that handles the lockstep execution of multiple ODEs
- `PerODE`: Memory layout where each ODE's variables are stored contiguously
- `PerIndex`: Memory layout where variables of the same index across ODEs are stored contiguously

### Constructor

```julia
LockstepFunction(f, u0, num_odes; internal_threading=true, ordering=PerODE())
```

- `f`: ODE function with signature `f(du, u, p, t)`
- `u0`: Batched initial condition vector for all ODEs
- `num_odes`: Number of ODE systems to solve
- `internal_threading`: Whether to use threading for parallel execution (default: `true`)
- `ordering`: Memory layout (`PerODE()` or `PerIndex()`, default: `PerODE()`)

### Main Functions

- `batch_initial_conditions(lockstep_func, u0)`: Batch initial conditions for all ODEs
- `batch_parameters(lockstep_func, p)`: Batch parameters for all ODEs  
- `extract_solutions(lockstep_func, sol)`: Extract individual ODE solutions from batched solution

### Standard Workflow

1. Create a `LockstepFunction`
2. Use it with standard `OrdinaryDiffEq.jl` functions:
   ```julia
   lockstep_func = LockstepFunction(my_ode!, u0_batched, num_odes)
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
