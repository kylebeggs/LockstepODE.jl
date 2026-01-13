# Callbacks

One of the powerful features of LockstepODE.jl is the ability to specify different callbacks for each ODE in your system. This is particularly useful when you want different behavior or conditions to trigger for different ODE instances while still benefiting from efficient parallel solving.

## Example 1: Different Reset Thresholds Per ODE

### Problem Setup

Consider solving multiple instances of an exponential growth model in parallel:

```math
\frac{du}{dt} = p \cdot u, \quad u(0) = 1
```

where `p` is the growth rate parameter. We want to solve three instances with `p = 1.0` over the time span t in [0, 5], but with different callback behaviors:

- **ODE 1**: Reset u to 1.0 when u > 3.0
- **ODE 2**: Reset u to 1.0 when u > 6.0
- **ODE 3**: Reset u to 1.0 when u > 10.0

### Implementation

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
lf = LockstepFunction(exponential_growth!, 1, 3; callbacks=[cb1, cb2, cb3])

# Create problem with per-ODE initial conditions and parameters
u0s = [[1.0], [1.0], [1.0]]
ps = [1.0, 1.0, 1.0]

prob = LockstepProblem(lf, u0s, (0.0, 5.0), ps)
sol = solve(prob, Tsit5())

# Access individual solutions directly
for i in 1:3
    println("  ODE $i: u=$(round(sol[i].u[end][1], digits=3)), resets=$(callback_counts[i])")
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
  ODE 1: u=2.991, resets=4
  ODE 2: u=5.983, resets=2
  ODE 3: u=2.447, resets=1
```

---

## Example 2: Shared Callback Applied to All ODEs

### Problem Setup

Sometimes you want the same callback behavior for all ODEs in your system. Consider the same exponential growth model with a single shared callback condition.

### Implementation

```julia
shared_count = Ref(0)
shared_cb = DiscreteCallback(
    (u, t, integrator) -> u[1] > 5.0,
    integrator -> (shared_count[] += 1; integrator.u[1] = 1.0)
)

# Pass a single callback - it applies to all ODEs
lf_shared = LockstepFunction(exponential_growth!, 1, 3; callbacks=shared_cb)

u0s = [[1.0], [1.0], [1.0]]
ps = [1.0, 1.0, 1.0]

prob_shared = LockstepProblem(lf_shared, u0s, (0.0, 5.0), ps)
sol_shared = solve(prob_shared, Tsit5())

for i in 1:3
    println("  ODE $i: u=$(round(sol_shared[i].u[end][1], digits=3))")
end
println("  Total resets: $(shared_count[])")
```

### Discussion

When you pass a single callback (not in a vector), LockstepODE applies it to all ODEs in the system. This is useful for:
- Enforcing global constraints across all ODEs
- Implementing shared stopping conditions
- Applying uniform physical constraints

### Expected Output

```
  ODE 1: u=4.978
  ODE 2: u=4.978
  ODE 3: u=4.978
  Total resets: 6
```

---

## Example 3: Different Parameters with Callbacks

### Problem Setup

Combine different parameters for each ODE with different callback behaviors:

### Implementation

```julia
growth_rates = [0.5, 1.0, 1.5]
thresholds = [4.0, 5.0, 6.0]
callback_logs = [Float64[] for _ in 1:3]

# Create per-ODE callbacks with different thresholds and logging
callbacks_varied = [
    DiscreteCallback(
        (u, t, integrator) -> u[1] > thresholds[i],
        integrator -> (push!(callback_logs[i], integrator.t); integrator.u[1] = 1.0)
    ) for i in 1:3
]

lf_varied = LockstepFunction(exponential_growth!, 1, 3; callbacks=callbacks_varied)

u0s = [[1.0], [1.0], [1.0]]
prob_varied = LockstepProblem(lf_varied, u0s, (0.0, 5.0), growth_rates)
sol_varied = solve(prob_varied, Tsit5())

for i in 1:3
    println("  ODE $i (rate=$(growth_rates[i]), thresh=$(thresholds[i])): " *
            "u=$(round(sol_varied[i].u[end][1], digits=3)), resets=$(length(callback_logs[i]))")
end
```

### Discussion

This example demonstrates the full flexibility of LockstepODE's per-ODE callback system:

1. **Different parameters**: Each ODE has a different growth rate
2. **Different callbacks**: Each callback has a different threshold
3. **Individual tracking**: Each callback logs to its own array

### Expected Output

```
  ODE 1 (rate=0.5, thresh=4.0): u=3.993, resets=1
  ODE 2 (rate=1.0, thresh=5.0): u=4.975, resets=2
  ODE 3 (rate=1.5, thresh=6.0): u=4.016, resets=2
```

---

## Callbacks in Both Modes

Callbacks work in both Batched and Ensemble modes:

### Batched Mode (Default)

Callbacks are automatically wrapped to work with the batched state:

```julia
prob = LockstepProblem(lf, u0s, tspan, ps)  # Default is Batched
sol = solve(prob, Tsit5())  # Callbacks work!
```

The callback wrapping is transparent - you use the same callback definitions.

### Ensemble Mode

Each ODE gets its own integrator with standard OrdinaryDiffEq callbacks:

```julia
prob = LockstepProblem{Ensemble}(lf, u0s, tspan, ps)
sol = solve(prob, Tsit5())  # Callbacks work directly
```

---

## ContinuousCallback Example

LockstepODE also supports `ContinuousCallback` for root-finding events:

```julia
using OrdinaryDiffEq: ContinuousCallback

# Bouncing ball - reverse velocity when position reaches zero
function ball!(du, u, p, t)
    du[1] = u[2]      # position
    du[2] = -9.81     # velocity (gravity)
end

# Continuous callback for ground detection
bounce_cb = ContinuousCallback(
    (u, t, integrator) -> u[1],  # Condition: position = 0
    integrator -> (integrator.u[2] = -0.8 * integrator.u[2])  # Affect: bounce
)

lf = LockstepFunction(ball!, 2, 3; callbacks=bounce_cb)
u0s = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]  # Different drop heights

prob = LockstepProblem(lf, u0s, (0.0, 5.0))
sol = solve(prob, Tsit5())
```

---

## Summary

LockstepODE.jl provides flexible callback support:

- **Per-ODE callbacks**: Pass a vector of callbacks to apply different conditions to each ODE
- **Shared callbacks**: Pass a single callback to apply the same behavior to all ODEs
- **Both callback types**: `DiscreteCallback` and `ContinuousCallback` are supported
- **Both modes**: Works in Ensemble and Batched modes transparently
- **Standard syntax**: Use the same callback definitions as standard OrdinaryDiffEq

This design pattern extends to all SciML callback types, making LockstepODE a powerful tool for parallel ODE solving with complex event handling.
