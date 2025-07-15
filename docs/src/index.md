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

LockstepODE.jl is a simple package that enables solving of multiple ordinary differential equation (ODE) systems in lockstep so that all ODEs are always at the same solution time with one another. The trick is it treats all the ODEs as one large system, so we do get penalized if one ODE has particularly bad parameters that require an adaptive scheme to crank the time step size really small, the entire system must adhere to this. We plan to address this limitation in the future. If your use case does not require all the ODEs to maintain the same solution time as one another, I would advise to use another package such within the SciML ecosystem. these situations are already covered.

## Features

- **SciML integration**: Fully compatible with the DifferentialEquations.jl ecosystem
- **Flexible memory layouts**: Choose between `PerODE` (default) and `PerIndex` ordering for optimal performance
- **Thread-based parallelism**: Built-in support for multi-threading via OhMyThreads.jl
- **GPU support**: CUDA acceleration available via KernelAbstractions.jl

## Installation

```julia
using Pkg
Pkg.add("LockstepODE")
```

## Quick Example

```julia
using LockstepODE
using OrdinaryDiffEq

# Define your ODE function
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Create a LockstepFunction for 100 parallel Lorenz systems
lockstep_func = LockstepFunction(lorenz!, 3, 100)

# Set up initial conditions (can be same or different for each ODE)
u0 = [1.0, 0.0, 0.0]  # Single initial condition, will be replicated
u0_batched = batch_initial_conditions(u0, 100, 3)

# Set up parameters and time span
p = (10.0, 28.0, 8/3)
tspan = (0.0, 100.0)

# Create and solve the problem
prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
sol = solve(prob, Tsit5())

# Extract individual solutions
individual_solutions = extract_solutions(lockstep_func, sol)
```

## Next Steps

- Read the [Getting Started](getting_started.md) guide for a detailed walkthrough
- Consult the [API Reference](api.md) for detailed documentation