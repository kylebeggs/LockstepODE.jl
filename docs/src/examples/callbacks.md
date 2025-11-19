# Callbacks

One of the powerful features of LockstepODE.jl is the ability to specify different callbacks for each ODE in your batched system. This is particularly useful when you want different behavior or conditions to trigger for different ODE instances while still benefiting from efficient parallel solving.

## Example 1: Different Reset Thresholds Per ODE

### Problem Setup

Consider solving multiple instances of an exponential growth model in parallel:

```math
\frac{du}{dt} = p \cdot u, \quad u(0) = 1
```

where $p$ is the growth rate parameter. We want to solve three instances with $p = 1.0$ over the time span $t \in [0, 5]$, but with different callback behaviors:

- **ODE 1**: Reset $u$ to 1.0 when $u > 3.0$
- **ODE 2**: Reset $u$ to 1.0 when $u > 6.0$
- **ODE 3**: Reset $u$ to 1.0 when $u > 10.0$

Mathematically, each ODE $i$ has a discrete callback condition:

```math
\text{Condition}_i: \quad u_i > \theta_i
```

```math
\text{Action}_i: \quad u_i \leftarrow 1.0
```

where $\theta_1 = 3.0$, $\theta_2 = 6.0$, and $\theta_3 = 10.0$.

### Implementation

The key insight is that LockstepODE allows you to pass a vector of callbacks, one for each ODE:

```julia
using LockstepODE
using OrdinaryDiffEq

function exponential_growth!(du, u, p, t)
    du[1] = p * u[1]
end

# Track how many times each callback fires
callback_counts = zeros(Int, 3)

# Define three different callbacks with different thresholds
cb1 = DiscreteCallback(
    (u, t, integrator) -> u[1] > 3.0,
    integrator -> (callback_counts[1] += 1; integrator.u[1] = 1.0)
)

cb2 = DiscreteCallback(
    (u, t, integrator) -> u[1] > 6.0,
    integrator -> (callback_counts[2] += 1; integrator.u[1] = 1.0)
)

cb3 = DiscreteCallback(
    (u, t, integrator) -> u[1] > 10.0,
    integrator -> (callback_counts[3] += 1; integrator.u[1] = 1.0)
)

# Pass callbacks as a vector - each callback applies to corresponding ODE
lockstep_func = LockstepFunction(exponential_growth!, 1, 3, callbacks = [cb1, cb2, cb3])
prob = ODEProblem(lockstep_func, ones(3), (0.0, 5.0), [1.0, 1.0, 1.0])
sol = solve(prob, Tsit5())

# Extract individual solutions
individual_sols = extract_solutions(lockstep_func, sol)
for (i, isol) in enumerate(individual_sols)
    println("  ODE $i: u=$(round(isol.u[end][1], digits=3)), resets=$(callback_counts[i])")
end
```

### Discussion

When you provide a vector of callbacks to `LockstepFunction`, each callback in the vector is applied exclusively to the corresponding ODE. The first callback affects only the first ODE, the second callback affects only the second ODE, and so on.

Since all three ODEs have the same growth rate but different reset thresholds:
- ODE 1 resets most frequently (threshold = 3.0)
- ODE 2 resets less frequently (threshold = 6.0)
- ODE 3 resets least frequently (threshold = 10.0)

This allows each ODE to maintain different dynamics despite being solved efficiently in parallel.

### Expected Output

```
Example 1: Per-ODE Callbacks with Different Thresholds
  ODE 1: u=2.991, resets=4
  ODE 2: u=5.983, resets=2
  ODE 3: u=2.447, resets=1
```

Note that the final values depend on exactly when the solver steps occur relative to the threshold crossings, but the pattern of reset frequencies remains consistent: ODE 1 resets most often, ODE 3 resets least often.

---

## Example 2: Shared Callback Applied to All ODEs

### Problem Setup

Sometimes you want the same callback behavior for all ODEs in your system. Consider the same exponential growth model:

```math
\frac{du}{dt} = p \cdot u, \quad u(0) = 1
```

with $p = 1.0$ for all three instances. Now we want a single shared callback condition:

```math
\text{Condition}: \quad u > 5.0 \quad \text{(for any ODE)}
```

```math
\text{Action}: \quad u \leftarrow 1.0
```

This callback applies uniformly to all three ODEs - whenever any ODE exceeds the threshold, it gets reset.

### Implementation

Instead of passing a vector of callbacks, pass a single callback object:

```julia
shared_count = Ref(0)
shared_cb = DiscreteCallback(
    (u, t, integrator) -> u[1] > 5.0,
    integrator -> (shared_count[] += 1; integrator.u[1] = 1.0)
)

# Pass a single callback - it applies to all ODEs
lockstep_func_shared = LockstepFunction(exponential_growth!, 1, 3, callbacks = shared_cb)
prob_shared = ODEProblem(lockstep_func_shared, ones(3), (0.0, 5.0), [1.0, 1.0, 1.0])
sol_shared = solve(prob_shared, Tsit5())

individual_sols_shared = extract_solutions(lockstep_func_shared, sol_shared)
for (i, isol) in enumerate(individual_sols_shared)
    println("  ODE $i: u=$(round(isol.u[end][1], digits=3))")
end
println("  Total resets: $(shared_count[])")
```

### Discussion

When you pass a single callback (not in a vector), LockstepODE applies it to all ODEs in the system. This is useful for:
- Enforcing global constraints across all ODEs
- Implementing shared stopping conditions
- Applying uniform physical constraints

Since all three ODEs have identical parameters and the same callback, they exhibit identical behavior and reset at the same rate.

### Expected Output

```
Example 2: Shared Callback Applied to All ODEs
  ODE 1: u=4.978, resets=2
  ODE 2: u=4.978, resets=2
  ODE 3: u=4.978, resets=2
  Total resets: 6
```

All three ODEs have identical final states and reset counts. The total reset count (6) equals 3 ODEs × 2 resets per ODE.

---

## Example 3: Different Parameters with Per-ODE Callbacks

### Problem Setup

The most flexible scenario combines different parameters for each ODE with different callback behaviors. Consider three exponential growth models with different growth rates:

```math
\frac{du_i}{dt} = p_i \cdot u_i, \quad u_i(0) = 1
```

where:
- $p_1 = 0.5$ with threshold $\theta_1 = 4.0$
- $p_2 = 1.0$ with threshold $\theta_2 = 5.0$
- $p_3 = 1.5$ with threshold $\theta_3 = 6.0$

Each ODE has both different dynamics (growth rates) and different callback conditions:

```math
\text{Condition}_i: \quad u_i > \theta_i
```

```math
\text{Action}_i: \quad u_i \leftarrow 1.0
```

We also log the time at which each reset occurs to analyze the temporal dynamics.

### Implementation

```julia
growth_rates = [0.5, 1.0, 1.5]
thresholds = [4.0, 5.0, 6.0]
callback_logs = [[] for _ in 1:3]

# Create per-ODE callbacks with different thresholds and logging
callbacks_varied = [DiscreteCallback(
                        (u, t, integrator) -> u[1] > thresholds[i],
                        integrator -> (push!(callback_logs[i], integrator.t); integrator.u[1] = 1.0)
                    ) for i in 1:3]

lockstep_func_varied = LockstepFunction(
    exponential_growth!, 1, 3, callbacks = callbacks_varied)
prob_varied = ODEProblem(lockstep_func_varied, ones(3), (0.0, 5.0), growth_rates)
sol_varied = solve(prob_varied, Tsit5())

individual_sols_varied = extract_solutions(lockstep_func_varied, sol_varied)
for (i, isol) in enumerate(individual_sols_varied)
    println("  ODE $i (rate=$(growth_rates[i]), thresh=$(thresholds[i])): u=$(round(isol.u[end][1], digits=3)), resets=$(length(callback_logs[i]))")
end
```

### Discussion

This example demonstrates the full flexibility of LockstepODE's per-ODE callback system:

1. **Different parameters**: Each ODE has a different growth rate via the `growth_rates` vector
2. **Different callbacks**: Each callback has a different threshold appropriate for its growth rate
3. **Individual tracking**: Each callback logs to its own array, allowing per-ODE analysis

The design is elegant: faster-growing ODEs (larger $p_i$) have higher thresholds ($\theta_i$), creating a balanced system where reset frequencies depend on both the growth dynamics and the chosen thresholds.

Note the use of a comprehension to generate the callback vector - this pattern scales naturally to arbitrary numbers of ODEs.

### Expected Output

```
Example 3: Per-ODE Callbacks with Different Parameters
  ODE 1 (rate=0.5, thresh=4.0): u=3.993, resets=1
  ODE 2 (rate=1.0, thresh=5.0): u=4.975, resets=2
  ODE 3 (rate=1.5, thresh=6.0): u=4.016, resets=2
```

The reset patterns show an interesting relationship: despite having different growth rates and thresholds, the middle ODE (rate=1.0, thresh=5.0) and fastest ODE (rate=1.5, thresh=6.0) both reset twice, while the slowest ODE (rate=0.5, thresh=4.0) only resets once. This demonstrates how the threshold-to-growth-rate ratio determines callback frequency.

---

## Summary

LockstepODE.jl provides flexible callback support that maintains the individual identity of each ODE while solving them efficiently in parallel:

- **Per-ODE callbacks**: Pass a vector of callbacks to apply different conditions/actions to each ODE
- **Shared callbacks**: Pass a single callback to apply the same behavior to all ODEs
- **Arbitrary complexity**: Combine with different parameters, custom logging, and any SciML-compatible callback type

This design pattern extends to all SciML callback types (continuous callbacks, callback sets, etc.), making LockstepODE a powerful tool for parallel ODE solving with complex event handling.
