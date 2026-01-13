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
using ModelingToolkit: t_nounits as t, D_nounits as D, getu, mtkcompile

# Define Lotka-Volterra (predator-prey) system with default parameters
@parameters α = 1.5 β = 1.0 γ = 3.0 δ = 1.0  # Set default parameter values
@variables x(t) y(t)  # prey and predator populations

eqs = [
    D(x) ~ α * x - β * x * y,  # Prey growth and predation
    D(y) ~ δ * x * y - γ * y,   # Predator growth and death
]
@named lotka_volterra = ODESystem(eqs, t)
lotka_volterra = mtkcompile(lotka_volterra)  # Compile system (required for MTK v10+)

# Create symbolic accessors using standard MTK getu()
# Note: getu works on each individual integrator in v2.0
get_x = getu(lotka_volterra, x)
get_y = getu(lotka_volterra, y)

# Number of parallel ODEs
num_odes = 3

# Same threshold for all ODEs
thresholds = [2.0, 2.0, 2.0]
reset_counts = zeros(Int, num_odes)
reset_times = [Float64[] for _ in 1:num_odes]

# Create per-ODE callbacks using symbolic variable access via getu()
callbacks = [
    DiscreteCallback(
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
        for i in 1:num_odes
]

# Create LockstepFunction with callbacks
lf = LockstepFunction(lotka_volterra, num_odes; callbacks = callbacks)

# Initial conditions: different for each ODE to show per-ODE behavior
u0s = [[1.0, 1.0], [2.0, 1.5], [3.0, 2.0]]
tspan = (0.0, 10.0)

# Extract default parameters from the compiled system (MTK v11 API)
using ModelingToolkit: parameters, hasdefault, getdefault
param_syms = parameters(lotka_volterra)
p = [Float64(getdefault(ps)) for ps in param_syms]

# Solve with explicit parameter vector (same for all ODEs)
prob = LockstepProblem(lf, u0s, tspan, p)
sol = solve(prob, Tsit5())

# Display results
println("\nParameters Used:")
println("="^60)
println("All ODEs use defaults: α=1.5, β=1.0, γ=3.0, δ=1.0")

println("\nCallback Results:")
println("="^60)
for i in 1:num_odes
    println("ODE $i (threshold=$(thresholds[i])):")
    println("  Resets: $(reset_counts[i])")
    println("  Reset times: $(round.(reset_times[i], digits = 3))")
    isol = sol[i]
    println("  Final state: prey=$(round(isol.u[end][1], digits = 3)), predator=$(round(isol.u[end][2], digits = 3))")
end
