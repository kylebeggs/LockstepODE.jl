# Per-ODE Callbacks Example
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

lockstep_func = LockstepFunction(exponential_growth!, 1, 3, callbacks = [cb1, cb2, cb3])
prob = ODEProblem(lockstep_func, ones(3), (0.0, 5.0), [1.0, 1.0, 1.0])
sol = solve(prob, Tsit5())

individual_sols = extract_solutions(lockstep_func, sol)
for (i, isol) in enumerate(individual_sols)
    println("  ODE $i: u=$(round(isol.u[end][1], digits=3)), resets=$(callback_counts[i])")
end

## Example 2: Shared callback for all ODEs
println("\nExample 2: Shared Callback Applied to All ODEs")
shared_count = Ref(0)
shared_cb = DiscreteCallback(
    (u, t, integrator) -> u[1] > 5.0,
    integrator -> (shared_count[] += 1; integrator.u[1] = 1.0)
)

lockstep_func_shared = LockstepFunction(exponential_growth!, 1, 3, callbacks = shared_cb)
prob_shared = ODEProblem(lockstep_func_shared, ones(3), (0.0, 5.0), [1.0, 1.0, 1.0])
sol_shared = solve(prob_shared, Tsit5())

individual_sols_shared = extract_solutions(lockstep_func_shared, sol_shared)
for (i, isol) in enumerate(individual_sols_shared)
    println("  ODE $i: u=$(round(isol.u[end][1], digits=3))")
end
println("  Total resets: $(shared_count[])")

## Example 3: Different parameters with per-ODE callbacks
println("\nExample 3: Per-ODE Callbacks with Different Parameters")
growth_rates = [0.5, 1.0, 1.5]
thresholds = [4.0, 5.0, 6.0]
callback_logs = [[] for _ in 1:3]

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
