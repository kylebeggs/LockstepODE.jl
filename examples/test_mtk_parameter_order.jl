using Pkg
Pkg.activate(@__DIR__)
using LockstepODE
using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

# Define Lotka-Volterra (predator-prey) system with multiple parameters
@parameters α β γ δ  # α: prey growth, β: predation rate, γ: predator death, δ: predator growth from prey
@variables x(t) y(t)  # x: prey population, y: predator population

eqs = [
    D(x) ~ α * x - β * x * y,
    D(y) ~ δ * x * y - γ * y
]
@named lotka_volterra = ODESystem(eqs, t)

# Create LockstepFunction with multiple ODEs
num_odes = 3
lockstep_func = LockstepFunction(lotka_volterra, 2, num_odes)  # 2 variables (x, y)

# Print parameter ordering from MTK to verify
lotka_volterra_simplified = structural_simplify(lotka_volterra)
println("Parameter order from ModelingToolkit:")
println(parameters(lotka_volterra_simplified))
println()

# NEW API: Specify parameters symbolically - order doesn't matter!
# The package automatically transforms to MTK's canonical ordering
param_sets = [
    Dict(α=>1.5, β=>1.0, γ=>3.0, δ=>1.0),  # ODE 1: high prey growth
    Dict(α=>1.0, β=>2.0, γ=>2.0, δ=>1.5),  # ODE 2: high predation rate
    Dict(α=>1.0, β=>1.0, γ=>1.0, δ=>2.0)   # ODE 3: high predator growth from prey
]

# Initial conditions: [prey, predator] for each ODE
u0 = vcat([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
tspan = (0.0, 10.0)

println("Testing parameter ordering with symbolic parameters:")
println("(Order specified by user - automatically transformed internally)")
for (i, p) in enumerate(param_sets)
    println("ODE $i: α=$(p[α]), β=$(p[β]), γ=$(p[γ]), δ=$(p[δ])")
end
println()

# Solve using symbolic parameters - no need to worry about MTK's internal ordering!
prob = ODEProblem(lockstep_func, u0, tspan, param_sets)
sol = solve(prob, Tsit5())

# Verify results by solving each ODE individually and comparing
println("Verification - comparing batched vs individual solutions:")
println()

for i in 1:num_odes
    # Solve individual ODE with symbolic parameters
    individual_prob = ODEProblem(
        lotka_volterra_simplified, [1.0, 1.0], tspan, param_sets[i])
    individual_sol = solve(individual_prob, Tsit5())

    # Extract from batched solution
    prey_idx = 2 * (i - 1) + 1
    pred_idx = 2 * (i - 1) + 2
    batched_prey = sol.u[end][prey_idx]
    batched_pred = sol.u[end][pred_idx]

    individual_prey = individual_sol.u[end][1]
    individual_pred = individual_sol.u[end][2]

    # Check if they match (use rtol=1e-2 to account for numerical differences)
    prey_match = isapprox(batched_prey, individual_prey, rtol = 1e-2)
    pred_match = isapprox(batched_pred, individual_pred, rtol = 1e-2)

    println("ODE $i (α=$(param_sets[i][α]), β=$(param_sets[i][β]), γ=$(param_sets[i][γ]), δ=$(param_sets[i][δ])):")
    println("  Prey:     batched=$(round(batched_prey, digits=6)), individual=$(round(individual_prey, digits=6)), match=$prey_match")
    println("  Predator: batched=$(round(batched_pred, digits=6)), individual=$(round(individual_pred, digits=6)), match=$pred_match")

    if !prey_match || !pred_match
        println("  ⚠️  MISMATCH DETECTED!")
    else
        println("  ✓ Parameters correctly handled")
    end
    println()
end

# Final summary
all_match = true
for i in 1:num_odes
    individual_prob = ODEProblem(
        lotka_volterra_simplified, [1.0, 1.0], tspan, param_sets[i])
    individual_sol = solve(individual_prob, Tsit5())

    prey_idx = 2 * (i - 1) + 1
    pred_idx = 2 * (i - 1) + 2

    if !isapprox(sol.u[end][prey_idx], individual_sol.u[end][1], rtol = 1e-2) ||
       !isapprox(sol.u[end][pred_idx], individual_sol.u[end][2], rtol = 1e-2)
        all_match = false
        break
    end
end

if all_match
    println("✓ SUCCESS: All batched solutions match individual solutions")
    println("Symbolic parameters API works correctly!")
else
    println("✗ FAILURE: Solutions do not match")
end
