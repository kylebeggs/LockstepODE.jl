# MTK Callbacks Verification Example
#
# This example includes detailed verification output to demonstrate that
# MTK's getu() correctly accesses per-ODE state in callbacks.
#
# Each callback prints:
# - Values returned by getu(x) and getu(y)
# - SubIntegrator's view of state
# - Full batched state array
# - Verification that getu() extracted the correct ODE's values
#
# For a clean example without verification output, see mtk_callbacks.jl

using Pkg
Pkg.activate(@__DIR__)
using LockstepODE
using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D, getu

# Define Lotka-Volterra (predator-prey) system
@parameters α β γ δ
@variables x(t) y(t)  # prey and predator populations

eqs = [
    D(x) ~ α * x - β * x * y,  # Prey growth and predation
    D(y) ~ δ * x * y - γ * y   # Predator growth and death
]
@named lotka_volterra = ODESystem(eqs, t)

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
                     # VERIFICATION: Show that getu() returns the correct ODE's values
                     println("\n" * "="^70)
                     println("ODE $i callback triggered at t=$(round(integrator.t, digits=3))")
                     println("-"^70)

                     # Show what getu returns
                     getu_x = get_x(integrator)
                     getu_y = get_y(integrator)
                     println("getu() returns:")
                     println("  x (prey)    = $getu_x")
                     println("  y (predator) = $getu_y")

                     # Show the SubIntegrator's view
                     println("\nSubIntegrator.u (view for ODE $i only):")
                     println("  [1] = $(integrator.u[1])  (should be x for ODE $i)")
                     println("  [2] = $(integrator.u[2])  (should be y for ODE $i)")

                     # Show the full batched state
                     full_state = integrator.parent.u
                     println("\nFull batched state [x₁, y₁, x₂, y₂, x₃, y₃]:")
                     for j in 1:num_odes
                         prey_pos = 2*(j-1) + 1
                         pred_pos = 2*(j-1) + 2
                         marker = (j == i) ? " ← ODE $i" : ""
                         println("  ODE $j: x=$(round(full_state[prey_pos], digits=4)), y=$(round(full_state[pred_pos], digits=4))$marker")
                     end

                     # VERIFY: getu matches the correct ODE's position in batched state
                     expected_prey_pos = 2*(i-1) + 1
                     expected_pred_pos = 2*(i-1) + 2
                     expected_x = full_state[expected_prey_pos]
                     expected_y = full_state[expected_pred_pos]

                     println("\nVERIFICATION:")
                     x_match = isapprox(getu_x, expected_x, rtol=1e-10)
                     y_match = isapprox(getu_y, expected_y, rtol=1e-10)
                     println("  getu(x) matches batched state position $expected_prey_pos? $(x_match ? "✓" : "✗")")
                     println("  getu(y) matches batched state position $expected_pred_pos? $(y_match ? "✓" : "✗")")

                     if !x_match || !y_match
                         println("\n  ⚠️  ERROR: getu() returned WRONG ODE's values!")
                     else
                         println("\n  ✓ CORRECT: getu() returned ODE $i's values")
                     end
                     println("="^70)

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
params = [
    Dict(α => 1.5, β => 1.0, γ => 3.0, δ => 1.0),  # ODE 1
    Dict(α => 1.0, β => 2.0, γ => 2.0, δ => 1.5),  # ODE 2
    Dict(α => 1.0, β => 1.0, γ => 1.0, δ => 2.0)   # ODE 3
]

# Initial conditions: different for each ODE to show per-ODE behavior
u0 = vcat([1.0, 1.0], [2.0, 1.5], [3.0, 2.0])
tspan = (0.0, 10.0)

# Solve
prob = ODEProblem(lockstep_func_with_cb, u0, tspan, params)
sol = solve(prob, Tsit5())

# Display results
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
