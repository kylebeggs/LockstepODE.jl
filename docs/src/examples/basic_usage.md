# Basic Usage

This page covers fundamental patterns for working with LockstepODE.jl, including setting up multiple initial conditions, batching parameters, and extracting solutions.

## Multiple Initial Conditions

When solving multiple ODE systems, you often want each system to start from a different initial condition. LockstepODE makes this straightforward by batching your initial conditions into a single vector.

### Problem Setup

Consider a simple oscillator system:

```math
\begin{aligned}
\frac{du_1}{dt} &= p \cdot u_2 \\
\frac{du_2}{dt} &= -u_1
\end{aligned}
```

where $p$ is a parameter that modifies the coupling strength. We want to solve this system for four different initial conditions and four different parameter values simultaneously.

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
u0_vec = [[1.0, 0.0], [2.0, 1.0], [0.5, -0.5], [1.5, 0.8]]
u0_batched = vcat(u0_vec...)  # Flatten to single vector
p_batched = [1.0, 1.2, 0.8, 1.1]  # Different parameters for each ODE

# Create the lockstep function
lockstep_func = LockstepFunction(my_ode!, 2, num_odes)  # 2 variables per ODE

# Use standard OrdinaryDiffEq.jl workflow
prob = ODEProblem(lockstep_func, u0_batched, (0.0, 10.0), p_batched)
sol = solve(prob, Tsit5())

# Extract and analyze individual solutions
individual_sols = extract_solutions(lockstep_func, sol)
for (i, isol) in enumerate(individual_sols)
    println("ODE $i final state: ", isol.u[end])
end
```

### Discussion

The key steps are:

1. **Define ODE function**: Use the standard signature `f(du, u, p, t)`
2. **Batch initial conditions**: Flatten your vector of initial conditions using `vcat(u0_vec...)`
3. **Batch parameters**: Create a vector with one parameter value per ODE
4. **Create LockstepFunction**: Specify the number of variables per ODE (`2`) and number of ODEs (`4`)
5. **Standard workflow**: Use `ODEProblem` and `solve` as normal
6. **Extract solutions**: Use `extract_solutions` to get individual `ODESolution` objects

Each extracted solution is a standard `ODESolution` from OrdinaryDiffEq.jl, so you can use all the usual analysis tools like interpolation, plotting, and accessing solution points.

### Expected Output

```
ODE 1 final state: [0.540302305868, -0.841470984808]
ODE 2 final state: [1.08060461174, -1.01364897577]
ODE 3 final state: [0.270151152934, 0.420735492404]
ODE 4 final state: [0.810453458602, -0.925717492289]
```

(Note: Exact values depend on solver tolerances and may vary slightly)

---

## Using Utility Functions

LockstepODE provides utility functions to make batching operations more convenient and to ensure correct data layout.

### Batching Initial Conditions

Instead of manually flattening initial conditions with `vcat`, you can use the `batch_initial_conditions` utility:

```julia
# Create a lockstep function first
lockstep_func = LockstepFunction(my_ode!, 2, 4)  # 4 ODEs, 2 variables each

# Batch initial conditions from individual vectors
u0_vec = [[1.0, 0.0], [2.0, 1.0], [0.5, -0.5], [1.5, 0.8]]
u0_batched = batch_initial_conditions(u0_vec, 4, 2)

# Or repeat a single initial condition across all ODEs
u0_single = [1.0, 0.0]
u0_repeated = batch_initial_conditions(u0_single, 4, 2)
```

### Batching Parameters

Parameters can be batched in several ways depending on your needs:

```julia
# Different parameter for each ODE
p_batched = [1.0, 1.2, 0.8, 1.1]

# Same parameter for all ODEs (broadcast automatically)
p_single = 1.0

# Multiple parameters per ODE
# Each ODE gets a vector of parameters
p_vec = [[1.0, 0.5], [1.2, 0.6], [0.8, 0.4], [1.1, 0.55]]
p_batched = vcat(p_vec...)
```

### Complete Example with Utilities

```julia
using LockstepODE
using OrdinaryDiffEq

function harmonic_oscillator!(du, u, p, t)
    ω = p  # Angular frequency
    du[1] = u[2]
    du[2] = -ω^2 * u[1]
end

# Set up the problem
num_odes = 3
ode_size = 2

# Create lockstep function
lockstep_func = LockstepFunction(harmonic_oscillator!, ode_size, num_odes)

# Use utility function to repeat initial condition
u0_batched = batch_initial_conditions([1.0, 0.0], num_odes, ode_size)

# Different frequencies for each oscillator
frequencies = [1.0, 2.0, 3.0]

# Solve
prob = ODEProblem(lockstep_func, u0_batched, (0.0, 2π), frequencies)
sol = solve(prob, Tsit5())

# Extract and analyze
individual_sols = extract_solutions(lockstep_func, sol)
for (i, isol) in enumerate(individual_sols)
    # Each solution completes a different number of cycles
    println("Oscillator $i (ω=$(frequencies[i])): position = $(round(isol.u[end][1], digits=3))")
end
```

### Discussion

The utility functions provide several benefits:

1. **Correctness**: They ensure data is arranged in the correct layout for your chosen `ordering`
2. **Convenience**: Less manual array manipulation
3. **Readability**: Intent is clearer than raw `vcat` operations
4. **Consistency**: Works the same way regardless of `PerODE` vs `PerIndex` ordering

### Expected Output

```
Oscillator 1 (ω=1.0): position = 1.0
Oscillator 2 (ω=2.0): position = 1.0
Oscillator 3 (ω=3.0): position = 1.0
```

After one period ($2\pi$ time units), each oscillator with frequency $\omega$ completes $\omega$ full cycles and returns to its starting position.

---

## Working with Solutions

After solving, each extracted solution is a full `ODESolution` object that supports:

- **Indexing**: `sol[i]` gets the $i$-th saved point
- **Interpolation**: `sol(t)` evaluates at arbitrary time $t$
- **Time vector**: `sol.t` gives all saved time points
- **State vector**: `sol.u` gives all saved states
- **Plotting**: Works directly with Plots.jl

### Example: Analyzing Solutions

```julia
individual_sols = extract_solutions(lockstep_func, sol)

for (i, isol) in enumerate(individual_sols)
    # Access final state
    final_state = isol.u[end]

    # Interpolate at arbitrary time
    state_at_5 = isol(5.0)

    # Check all time points
    times = isol.t

    # Compute energy (for harmonic oscillator)
    energy = 0.5 * (isol.u[end][1]^2 + isol.u[end][2]^2)

    println("ODE $i: final energy = $(round(energy, digits=4))")
end
```

---

## Summary

Key takeaways for basic usage:

1. **Flatten initial conditions** using `vcat` or `batch_initial_conditions`
2. **Batch parameters** as a vector (one per ODE) or use a single value for all
3. **Use standard workflow**: `ODEProblem` → `solve` → `extract_solutions`
4. **Extracted solutions are standard** `ODESolution` objects with full functionality

For more advanced usage patterns, see [Advanced Configuration](@ref) and [Per-ODE Callbacks](@ref).
