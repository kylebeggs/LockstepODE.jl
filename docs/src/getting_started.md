# Getting Started

This guide will walk you through using LockstepODE.jl to solve multiple ODEs in parallel.

## Basic Workflow

The typical workflow with LockstepODE consists of four steps:

1. Define your ODE function (same as standard DifferentialEquations.jl)
2. Create a `LockstepFunction` wrapper
3. Prepare batched initial conditions
4. Solve and extract individual solutions

## Step 1: Define Your ODE Function

Your ODE function should follow the standard in-place form expected by DifferentialEquations.jl:

```julia
function my_ode!(du, u, p, t)
    # Your ODE equations here
    du[1] = ...
    du[2] = ...
end
```

## Step 2: Create a LockstepFunction

Wrap your ODE function with `LockstepFunction`, specifying the size of each ODE and how many to solve:

```julia
lockstep_func = LockstepFunction(
    my_ode!,           # Your ODE function
    ode_size,          # Number of variables per ODE
    num_odes;          # Number of ODEs to solve in parallel
    internal_threading = true,  # Enable threading (default)
    ordering = PerODE()        # Memory layout (default)
)
```

## Step 3: Prepare Initial Conditions

LockstepODE provides flexible ways to specify initial conditions:

### Same Initial Conditions for All ODEs

```julia
u0_single = [1.0, 2.0, 3.0]  # Initial conditions for one ODE
u0_batched = batch_initial_conditions(u0_single, num_odes, ode_size)
```

### Different Initial Conditions for Each ODE

```julia
# Vector of initial condition vectors
u0_multiple = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
u0_batched = batch_initial_conditions(u0_multiple, num_odes, ode_size)
```

### Pre-batched Initial Conditions

```julia
# If you already have a flat vector of all initial conditions
u0_batched = ones(num_odes * ode_size)
```

## Step 4: Solve and Extract Solutions

```julia
# Create the ODE problem
prob = ODEProblem(lockstep_func, u0_batched, tspan, p)

# Solve using any DifferentialEquations.jl solver
sol = solve(prob, Tsit5())

# Extract individual solutions
individual_solutions = extract_solutions(lockstep_func, sol)

# Access specific solutions
for (i, sol_i) in enumerate(individual_solutions)
    println("ODE $i final state: ", sol_i.u[end])
end
```

## Complete Example: Exponential Decay

Here's a complete example solving multiple exponential decay ODEs with different decay rates:

```julia
using LockstepODE
using OrdinaryDiffEq
using Plots

# Define the ODE
function exponential_decay!(du, u, p, t)
    du[1] = -p * u[1]
end

# Setup parameters
num_odes = 10
ode_size = 1
decay_rates = range(0.1, 2.0, length=num_odes)

# Create LockstepFunction
lockstep_func = LockstepFunction(exponential_decay!, ode_size, num_odes)

# Initial conditions (all start at 1.0)
u0 = [1.0]
u0_batched = batch_initial_conditions(u0, num_odes, ode_size)

# Time span
tspan = (0.0, 5.0)

# Create and solve the problem
# Note: parameters can be a vector (one per ODE) or scalar (shared)
prob = ODEProblem(lockstep_func, u0_batched, tspan, decay_rates)
sol = solve(prob, Tsit5())

# Extract and plot individual solutions
individual_solutions = extract_solutions(lockstep_func, sol)

plot()
for (i, sol_i) in enumerate(individual_solutions)
    plot!(sol_i.t, [u[1] for u in sol_i.u], 
          label="λ = $(decay_rates[i])", 
          lw=2)
end
xlabel!("Time")
ylabel!("Value")
title!("Exponential Decay with Different Rates")
```

## Memory Ordering

LockstepODE supports two memory layouts:

### PerODE (Default)
Variables for each ODE are stored contiguously. This is usually the best choice:
```julia
lockstep_func = LockstepFunction(my_ode!, ode_size, num_odes, ordering=PerODE())
```

### PerIndex
Variables of the same index across ODEs are stored contiguously:
```julia
lockstep_func = LockstepFunction(my_ode!, ode_size, num_odes, ordering=PerIndex())
```

## Threading Control

By default, LockstepODE uses internal threading. You can disable this if you want to control threading externally:

```julia
lockstep_func = LockstepFunction(my_ode!, ode_size, num_odes, internal_threading=false)
```

## Parameter Handling

Parameters can be specified in two ways:

1. **Shared parameters**: All ODEs use the same parameters
   ```julia
   p = (1.0, 2.0, 3.0)  # Tuple or scalar
   prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
   ```

2. **Per-ODE parameters**: Each ODE has its own parameters
   ```julia
   p = [p1, p2, p3, ...]  # Vector with one element per ODE
   prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
   ```

## GPU Acceleration

LockstepODE.jl supports multiple GPU backends through automatic array-type dispatch. Simply use GPU arrays as initial conditions, and the appropriate GPU kernel will be used automatically.

### Supported GPU Backends

#### NVIDIA GPUs (CUDA)
```julia
using LockstepODE
using CUDA  # Activates CUDA extension

u0_batched = CuArray(u0_batched)  # Move to GPU
prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
sol = solve(prob, Tsit5())  # Automatically uses CUDA kernel
```

#### AMD GPUs (ROCm)
```julia
using LockstepODE
using AMDGPU  # Activates AMDGPU extension

u0_batched = ROCArray(u0_batched)  # Move to AMD GPU
prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
sol = solve(prob, Tsit5())  # Automatically uses ROCm kernel
```

#### Apple Silicon (Metal)
```julia
using LockstepODE
using Metal  # Activates Metal extension

u0_batched = MtlArray(u0_batched)  # Move to Metal GPU
prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
sol = solve(prob, Tsit5())  # Automatically uses Metal kernel
```

#### Intel GPUs (oneAPI)
```julia
using LockstepODE
using oneAPI  # Activates oneAPI extension

u0_batched = oneArray(u0_batched)  # Move to Intel GPU
prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
sol = solve(prob, Tsit5())  # Automatically uses oneAPI kernel
```

### GPU Notes

- GPU backends are optional: only install the ones you need
- Backend selection is automatic based on array type
- All backends use the same `LockstepFunction` - no code changes needed
- GPU acceleration is most beneficial for large numbers of ODEs (100+)