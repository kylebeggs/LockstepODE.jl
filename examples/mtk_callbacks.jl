# MTK Callbacks Example
#
# Demonstrates using ModelingToolkit's symbolic variable access (getu)
# in per-ODE callbacks for parallel Lotka-Volterra systems.
#
# Key features:
# - Multi-equation system (prey x, predator y)
# - Multiple parameters (α, β, γ, δ)
# - Per-ODE callbacks with different behavior
# - Symbolic variable access via getu() in callbacks

using Pkg
Pkg.activate(@__DIR__)
using LockstepODE
using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D, getu

# Define Lotka-Volterra (predator-prey) system with default parameters
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0  # Set default parameter values
@variables x(t) y(t)  # prey and predator populations

eqs = [
    D(x) ~ α * x - β * x * y,  # Prey growth and predation
    D(y) ~ δ * x * y - γ * y   # Predator growth and death
]
@named lotka_volterra = ODESystem(eqs, t)
lotka_volterra = complete(lotka_volterra)  # Complete system to enable partial parameter specification

# Create symbolic accessors using standard MTK getu()
get_x = getu(lotka_volterra, x)
get_y = getu(lotka_volterra, y)

# Create LockstepFunction for 3 parallel ODEs
num_odes = 3
lockstep_func = LockstepFunction(lotka_volterra, 2, num_odes)

# Same threshold for all ODEs to expose the getu bug
thresholds = [2.0, 2.0, 2.0]
reset_counts = zeros(Int, num_odes)
reset_times = [Float64[] for _ in 1:num_odes]

# Create per-ODE callbacks using symbolic variable access via getu()
callbacks = [DiscreteCallback(
                 # Condition: check if predator (y) exceeds threshold using symbolic accessor
                 (u, t, integrator) -> get_y(integrator) > thresholds[i],
                 # Affect: reset both prey and predator populations and track
                 integrator -> begin
                     reset_counts[i] += 1
                     push!(reset_times[i], integrator.t)
                     integrator.u[1] = 1.0  # Reset prey (x)
                     integrator.u[2] = 1.0  # Reset predator (y)
                 end
             )
             for i in 1:num_odes]

# Create LockstepFunction with callbacks
lockstep_func_with_cb = LockstepFunction(
    lotka_volterra, 2, num_odes, callbacks = callbacks
)

# Specify parameters symbolically for each ODE
# Partial specification supported: only provide changed parameters, rest use defaults
params = [
    Dict(),                      # ODE 1: use all defaults (α=1.5, β=1.0, γ=3.0, δ=1.0)
    Dict(α => 1.0, β => 2.0, γ => 2.0, δ => 1.5),  # ODE 2: all specified
    Dict(β => 0.5, δ => 2.5)     # ODE 3: only β,δ changed, α,γ use defaults
]

# Initial conditions: different for each ODE to show per-ODE behavior
u0 = vcat([1.0, 1.0], [2.0, 1.5], [3.0, 2.0])
tspan = (0.0, 10.0)

# Solve
prob = ODEProblem(lockstep_func_with_cb, u0, tspan, params)
sol = solve(prob, Tsit5())

# Display results
println("\nParameters Used:")
println("="^60)
println("ODE 1: α=1.5 (default), β=1.0 (default), γ=3.0 (default), δ=1.0 (default)")
println("ODE 2: α=1.0, β=2.0, γ=2.0, δ=1.5")
println("ODE 3: α=1.5 (default), β=0.5, γ=3.0 (default), δ=2.5")

println("\nCallback Results:")
println("="^60)
for i in 1:num_odes
    println("ODE $i (threshold=$(thresholds[i])):")
    println("  Resets: $(reset_counts[i])")
    println("  Reset times: $(round.(reset_times[i], digits=3))")
    prey_idx = 2 * (i - 1) + 1
    pred_idx = 2 * (i - 1) + 2
    println("  Final state: prey=$(round(sol.u[end][prey_idx], digits=3)), predator=$(round(sol.u[end][pred_idx], digits=3))")
end
