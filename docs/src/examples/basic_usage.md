# Basic Usage

This page covers fundamental patterns for working with LockstepODE.jl v2.0, including setting up multiple initial conditions, handling parameters, and accessing solutions.

## Multiple Initial Conditions

When solving multiple ODE systems, you often want each system to start from a different initial condition. LockstepODE makes this straightforward with automatic normalization.

### Problem Setup

Consider a simple oscillator system:

```math
\begin{aligned}
\frac{du_1}{dt} &= p \cdot u_2 \\
\frac{du_2}{dt} &= -u_1
\end{aligned}
```

where `p` is a parameter that modifies the coupling strength. We want to solve this system for four different initial conditions and four different parameter values simultaneously.

### Implementation

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
u0s = [[1.0, 0.0], [2.0, 1.0], [0.5, -0.5], [1.5, 0.8]]
ps = [1.0, 1.2, 0.8, 1.1]  # Different parameters for each ODE

# Create the LockstepFunction
lf = LockstepFunction(my_ode!, 2, num_odes)  # 2 variables per ODE

# Create problem and solve
prob = LockstepProblem(lf, u0s, (0.0, 10.0), ps)
sol = solve(prob, Tsit5())

# Access individual solutions directly
for i in 1:num_odes
    println("ODE $i final state: ", sol[i].u[end])
end
```

### Discussion

The key steps in v2.0 are:

1. **Define ODE function**: Use the standard signature `f(du, u, p, t)`
2. **Prepare initial conditions**: Vector of vectors (one per ODE)
3. **Prepare parameters**: Vector of parameters (one per ODE) or single shared value
4. **Create LockstepFunction**: Specify ODE size and count
5. **Create LockstepProblem**: Pass initial conditions, tspan, and parameters
6. **Solve and access**: Use `sol[i]` to access individual solutions

Each solution supports interpolation, so you can do `sol[i](5.0)` to get the state at any time.

---

## Execution Modes

LockstepODE v2.0 supports two execution modes:

### Ensemble Mode (Default)

N independent integrators, each with adaptive timestepping:

```julia
# Ensemble mode is the default
prob = LockstepProblem(lf, u0s, tspan, ps)
sol = solve(prob, Tsit5())

# Or explicitly
prob = LockstepProblem{Ensemble}(lf, u0s, tspan, ps)
```

**Use for:**
- Per-ODE introspection during integration
- Complex per-ODE callbacks
- ModelingToolkit systems
- When ODEs have very different dynamics

### Batched Mode

Single integrator with batched state and parallel RHS:

```julia
prob = LockstepProblem{Batched}(lf, u0s, tspan, ps;
    ordering = PerODE(),        # Memory layout
    internal_threading = true   # CPU threading
)
sol = solve(prob, Tsit5())
```

**Use for:**
- Large N (100+ ODEs)
- GPU acceleration
- Maximum performance

---

## Parameter Handling

Parameters are automatically normalized:

```julia
# Shared parameter (same for all ODEs)
prob = LockstepProblem(lf, u0s, tspan, 1.5)

# Per-ODE scalar parameters
prob = LockstepProblem(lf, u0s, tspan, [1.0, 1.2, 0.8])

# Per-ODE vector parameters
ps = [[1.0, 0.5], [1.2, 0.6], [0.8, 0.4]]
prob = LockstepProblem(lf, u0s, tspan, ps)
```

---

## Complete Example: Harmonic Oscillators

```julia
using LockstepODE
using OrdinaryDiffEq

function harmonic_oscillator!(du, u, p, t)
    w = p  # Angular frequency
    du[1] = u[2]
    du[2] = -w^2 * u[1]
end

# Set up
num_odes = 3
ode_size = 2

# Create LockstepFunction
lf = LockstepFunction(harmonic_oscillator!, ode_size, num_odes)

# Same initial condition for all
u0s = [[1.0, 0.0] for _ in 1:num_odes]

# Different frequencies
frequencies = [1.0, 2.0, 3.0]

# Solve
prob = LockstepProblem(lf, u0s, (0.0, 2*pi), frequencies)
sol = solve(prob, Tsit5())

# Analyze
for i in 1:num_odes
    pos = sol[i].u[end][1]
    println("Oscillator $i (w=$(frequencies[i])): final position = $(round(pos, digits=3))")
end
```

---

## CommonSolve Interface

LockstepODE implements the full CommonSolve.jl interface:

```julia
# Initialize without solving
integ = init(prob, Tsit5())

# Manual stepping
step!(integ)                    # One adaptive step
step!(integ, 0.1, true)         # Step by dt, stop at t+dt

# Access during integration
integ.t                         # Current time
integ.u                         # Vector of current states
integ[i]                        # i-th sub-integrator

# Complete the solve
sol = solve!(integ)

# Reinitialize with new conditions
new_u0s = [[0.5, 0.0], [1.5, 0.0], [2.5, 0.0]]
reinit!(integ, new_u0s)
```

---

## Working with Solutions

Each solution supports full access:

```julia
sol = solve(prob, Tsit5())

# Access individual ODE solutions
sol[i]              # i-th ODE solution
sol[i].u            # Time series of states
sol[i].t            # Time points
sol[i](t)           # Interpolate at arbitrary time t
sol[i].retcode      # Return code for this ODE

# Combined solution properties
sol.retcode         # Overall return code
length(sol)         # Number of ODEs

# Extract all states at a specific time
states = extract_at_time(sol, 5.0)
```

### Example: Analyzing Solutions

```julia
sol = solve(prob, Tsit5())

for i in 1:length(sol)
    # Access final state
    final_state = sol[i].u[end]

    # Interpolate at arbitrary time
    state_at_5 = sol[i](5.0)

    # Compute energy (for harmonic oscillator)
    energy = 0.5 * (final_state[1]^2 + final_state[2]^2)

    println("ODE $i: final energy = $(round(energy, digits=4))")
end
```

---

## Initial Condition Normalization

LockstepProblem automatically normalizes initial conditions:

```julia
# Single initial condition replicated for all ODEs
prob = LockstepProblem(lf, [1.0, 0.0], tspan)

# Vector of per-ODE initial conditions
prob = LockstepProblem(lf, [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], tspan)
```

---

## Summary

Key takeaways for v2.0:

1. **Use LockstepProblem** instead of ODEProblem
2. **Pass vector of vectors** for initial conditions (automatic normalization)
3. **Choose execution mode**: Ensemble (default) or Batched
4. **Access solutions directly** with `sol[i]` (no extract_solutions needed)
5. **CommonSolve interface**: `init`, `step!`, `solve!`, `reinit!`

For more advanced usage, see [Advanced Configuration](advanced_configuration.md) and [Callbacks](callbacks.md).
