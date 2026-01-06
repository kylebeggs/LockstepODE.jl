# Callbacks Example
# Demonstrates using different callbacks for different ODE systems

using LockstepODE
using OrdinaryDiffEq

function exponential_growth!(du, u, p, t)
    du[1] = p * u[1]
end

# Example 1: Different reset thresholds per ODE
println("Example 1: Per-ODE Callbacks with Different Thresholds")
callback_counts = zeros(Int, 3)

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

lf = LockstepFunction(exponential_growth!, 1, 3; callbacks = [cb1, cb2, cb3])
u0s = [[1.0], [1.0], [1.0]]
ps = [1.0, 1.0, 1.0]
prob = LockstepProblem(lf, u0s, (0.0, 5.0), ps)
sol = solve(prob, Tsit5())

for (i, isol) in enumerate(sol.solutions)
    println("  ODE $i: u=$(round(isol.u[end][1], digits=3)), resets=$(callback_counts[i])")
end

## Example 2: Shared callback for all ODEs
println("\nExample 2: Shared Callback Applied to All ODEs")
shared_count = Ref(0)
shared_cb = DiscreteCallback(
    (u, t, integrator) -> u[1] > 5.0,
    integrator -> (shared_count[] += 1; integrator.u[1] = 1.0)
)

lf_shared = LockstepFunction(exponential_growth!, 1, 3; callbacks = shared_cb)
prob_shared = LockstepProblem(lf_shared, u0s, (0.0, 5.0), ps)
sol_shared = solve(prob_shared, Tsit5())

for (i, isol) in enumerate(sol_shared.solutions)
    println("  ODE $i: u=$(round(isol.u[end][1], digits=3))")
end
println("  Total resets: $(shared_count[])")

## Example 3: Different parameters with per-ODE callbacks
println("\nExample 3: Per-ODE Callbacks with Different Parameters")
growth_rates = [0.5, 1.0, 1.5]
thresholds = [4.0, 5.0, 6.0]
callback_logs = [Float64[] for _ in 1:3]

callbacks_varied = [DiscreteCallback(
                        (u, t, integrator) -> u[1] > thresholds[i],
                        integrator -> (
                            push!(callback_logs[i], integrator.t); integrator.u[1] = 1.0)
                    ) for i in 1:3]

lf_varied = LockstepFunction(exponential_growth!, 1, 3; callbacks = callbacks_varied)
prob_varied = LockstepProblem(lf_varied, u0s, (0.0, 5.0), growth_rates)
sol_varied = solve(prob_varied, Tsit5())

for (i, isol) in enumerate(sol_varied.solutions)
    println("  ODE $i (rate=$(growth_rates[i]), thresh=$(thresholds[i])): u=$(round(isol.u[end][1], digits=3)), resets=$(length(callback_logs[i]))")
end
