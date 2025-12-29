# Getting Started

This guide walks you through using LockstepODE.jl to solve multiple ODEs in parallel.

## Basic Workflow

LockstepODE v2.0 uses a simple four-step workflow:

1. Define your ODE function (same as standard DifferentialEquations.jl)
2. Create a `LockstepFunction` wrapper
3. Create a `LockstepProblem` with initial conditions
4. Solve and access individual solutions

## Step 1: Define Your ODE Function

Your ODE function should follow the standard in-place form:

```julia
function my_ode!(du, u, p, t)
    du[1] = ...
    du[2] = ...
end
```

## Step 2: Create a LockstepFunction

Wrap your ODE function with `LockstepFunction`:

```julia
lf = LockstepFunction(
    my_ode!,    # Your ODE function
    ode_size,   # Number of variables per ODE
    num_odes;   # Number of ODEs to solve in parallel
    callbacks = nothing  # Optional per-ODE callbacks
)
```

## Step 3: Create a LockstepProblem

```julia
# Initial conditions as vector of vectors
u0s = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

# Create problem (Ensemble mode by default)
prob = LockstepProblem(lf, u0s, tspan, p)

# Or explicitly use Batched mode
prob = LockstepProblem{Batched}(lf, u0s, tspan, p)
```

## Step 4: Solve and Access Solutions

```julia
# Solve using any OrdinaryDiffEq solver
sol = solve(prob, Tsit5())

# Access individual solutions directly
sol[1]       # First ODE's solution
sol[2].u     # Second ODE's state time series
sol[3](1.5)  # Third ODE interpolated at t=1.5
```

## Execution Modes

LockstepODE provides two execution modes:

### Ensemble Mode (Default)

N independent ODE integrators, each with adaptive timestepping:

```julia
prob = LockstepProblem(lf, u0s, tspan, p)
# or explicitly:
prob = LockstepProblem{Ensemble}(lf, u0s, tspan, p)
```

**Best for:**
- Per-ODE introspection and control
- Complex per-ODE callbacks
- ModelingToolkit integration
- When ODEs have very different dynamics

### Batched Mode

Single integrator with batched state vector and parallel RHS evaluation:

```julia
prob = LockstepProblem{Batched}(lf, u0s, tspan, p;
    ordering = PerODE(),        # Memory layout
    internal_threading = true   # CPU threading
)
```

**Best for:**
- Large number of ODEs (N > 100)
- GPU acceleration
- Maximum performance when per-ODE control isn't needed

## Complete Example: Exponential Decay

```julia
using LockstepODE
using OrdinaryDiffEq

# Define the ODE
function exponential_decay!(du, u, p, t)
    du[1] = -p * u[1]
end

# Setup
num_odes = 10
ode_size = 1
decay_rates = range(0.1, 2.0, length=num_odes)

# Create LockstepFunction
lf = LockstepFunction(exponential_decay!, ode_size, num_odes)

# Initial conditions (all start at 1.0)
u0s = [[1.0] for _ in 1:num_odes]

# Create and solve
prob = LockstepProblem(lf, u0s, (0.0, 5.0), collect(decay_rates))
sol = solve(prob, Tsit5())

# Access solutions
for i in 1:num_odes
    println("ODE $i final state: ", sol[i].u[end])
end
```

## CommonSolve Interface

LockstepODE implements the full CommonSolve.jl interface:

```julia
# Initialize integrator without solving
integ = init(prob, Tsit5())

# Manual stepping
step!(integ)                    # One adaptive step
step!(integ, 0.1, true)         # Step by dt=0.1, stop exactly at t+dt

# Access during integration
integ.t                         # Current time
integ.u                         # Current states (vector of vectors)
integ[i]                        # i-th sub-integrator

# Complete the solve
sol = solve!(integ)

# Or solve directly
sol = solve(prob, Tsit5())

# Reinitialize with new conditions
reinit!(integ, new_u0s)
```

## Initial Condition Normalization

LockstepProblem automatically normalizes initial conditions:

```julia
# Single initial condition (replicated for all ODEs)
prob = LockstepProblem(lf, [1.0, 0.0], tspan)

# Vector of initial conditions (one per ODE)
prob = LockstepProblem(lf, [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], tspan)
```

## Parameter Handling

Parameters work the same way:

```julia
# Shared parameter for all ODEs
prob = LockstepProblem(lf, u0s, tspan, 0.5)

# Per-ODE parameters
prob = LockstepProblem(lf, u0s, tspan, [0.1, 0.5, 1.0])
```

## Memory Layouts (Batched Mode)

Batched mode supports two memory layouts:

### PerODE (Default)

Variables for each ODE stored contiguously:
```
[u1_ode1, u2_ode1, u1_ode2, u2_ode2, ...]
```

```julia
prob = LockstepProblem{Batched}(lf, u0s, tspan; ordering=PerODE())
```

### PerIndex

Same variable across ODEs stored contiguously:
```
[u1_ode1, u1_ode2, u2_ode1, u2_ode2, ...]
```

```julia
prob = LockstepProblem{Batched}(lf, u0s, tspan; ordering=PerIndex())
```

## GPU Acceleration

Use Batched mode with GPU arrays:

```julia
using LockstepODE
using CUDA  # Or: AMDGPU, Metal, oneAPI

# Create GPU initial conditions
u0s_gpu = [CuArray([1.0, 0.0]) for _ in 1:1000]

# Use Batched mode (required for GPU)
prob = LockstepProblem{Batched}(lf, u0s_gpu, tspan, p)
sol = solve(prob, Tsit5())  # Runs on GPU
```

### Supported Backends

| Package | GPU Type | Array Type |
|---------|----------|------------|
| CUDA.jl | NVIDIA | `CuArray` |
| AMDGPU.jl | AMD | `ROCArray` |
| Metal.jl | Apple Silicon | `MtlArray` |
| oneAPI.jl | Intel | `oneArray` |

### GPU Notes

- GPU backends are optional extensions - only install what you need
- Backend selection is automatic based on array type
- GPU acceleration is most beneficial for large N (100+ ODEs)
- All backends use the same code - just change the array type

## Solution Access

```julia
sol = solve(prob, Tsit5())

# Individual solution access
sol[i]           # i-th ODE solution
sol[i].u         # Time series of states
sol[i].t         # Time points
sol[i](t)        # Interpolate at time t
sol[i].retcode   # Return code

# Combined solution properties
sol.retcode      # Overall return code
length(sol)      # Number of ODEs

# Extract all states at a specific time
states = extract_at_time(sol, 5.0)
```

## Threading Control

Disable internal threading for external parallelization:

```julia
# Ensemble mode: threading controlled by Julia threads
# Batched mode: disable internal threading
prob = LockstepProblem{Batched}(lf, u0s, tspan; internal_threading=false)
```
