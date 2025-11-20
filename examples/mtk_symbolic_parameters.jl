using LockstepODE
using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

println("="^70)
println("MTK Symbolic Parameter API Demo")
println("="^70)
println()

# Define Lotka-Volterra (predator-prey) system
@parameters α β γ δ
@variables x(t) y(t)

eqs = [
    D(x) ~ α * x - β * x * y,  # Prey growth and predation
    D(y) ~ δ * x * y - γ * y   # Predator growth and death
]
@named lotka_volterra = ODESystem(eqs, t)

println("System defined:")
println("  dxdt = α*x - β*x*y")
println("  dydt = δ*x*y - γ*y")
println()

# Create LockstepFunction for 3 parallel ODEs
num_odes = 3
lockstep_func = LockstepFunction(lotka_volterra, 2, num_odes)

println("Created LockstepFunction for $num_odes parallel ODEs")
println()

# ============================================================================
# NEW API: Symbolic Parameters (Recommended)
# ============================================================================
println("─"^70)
println("Using Symbolic Parameters API (RECOMMENDED)")
println("─"^70)
println()

# Specify parameters using Dicts - order doesn't matter!
params_symbolic = [
    Dict(α=>1.5, β=>1.0, γ=>3.0, δ=>1.0),  # ODE 1
    Dict(α=>1.0, β=>2.0, γ=>2.0, δ=>1.5),  # ODE 2
    Dict(α=>1.0, β=>1.0, γ=>1.0, δ=>2.0)   # ODE 3
]

println("Parameters specified symbolically:")
for (i, p) in enumerate(params_symbolic)
    println("  ODE $i: α=$(p[α]), β=$(p[β]), γ=$(p[γ]), δ=$(p[δ])")
end
println()

u0 = vcat([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
tspan = (0.0, 10.0)

# Create ODEProblem - parameters are automatically transformed!
prob_symbolic = ODEProblem(lockstep_func, u0, tspan, params_symbolic)
sol_symbolic = solve(prob_symbolic, Tsit5())

println("Solved successfully!")
println("Final values:")
for i in 1:num_odes
    prey_idx = 2 * (i - 1) + 1
    pred_idx = 2 * (i - 1) + 2
    println("  ODE $i: prey=$(round(sol_symbolic.u[end][prey_idx], digits=4)), " *
            "predator=$(round(sol_symbolic.u[end][pred_idx], digits=4))")
end
println()

# ============================================================================
# OLD API: Flat Vector Parameters (Still Supported)
# ============================================================================
println("─"^70)
println("Using Flat Vector API (STILL SUPPORTED)")
println("─"^70)
println()

# Must manually check MTK's canonical parameter ordering
sys_simplified = structural_simplify(lotka_volterra)
canonical_order = parameters(sys_simplified)
println("MTK canonical parameter order: $canonical_order")
println()

# Must provide parameters in this specific order: [α, β, δ, γ]
params_flat = vcat(
    [1.5, 1.0, 1.0, 3.0],  # [α, β, δ, γ] for ODE 1
    [1.0, 2.0, 1.5, 2.0],  # [α, β, δ, γ] for ODE 2
    [1.0, 1.0, 2.0, 1.0]   # [α, β, δ, γ] for ODE 3
)

println("Parameters in flat vector (canonical order):")
println("  ", params_flat)
println()

prob_flat = ODEProblem(lockstep_func, u0, tspan, params_flat)
sol_flat = solve(prob_flat, Tsit5())

println("Solved successfully!")
println("Final values:")
for i in 1:num_odes
    prey_idx = 2 * (i - 1) + 1
    pred_idx = 2 * (i - 1) + 2
    println("  ODE $i: prey=$(round(sol_flat.u[end][prey_idx], digits=4)), " *
            "predator=$(round(sol_flat.u[end][pred_idx], digits=4))")
end
println()

# ============================================================================
# Verification
# ============================================================================
println("─"^70)
println("Verification: Both APIs produce identical results")
println("─"^70)
println()

all_match = true
for i in 1:num_odes
    prey_idx = 2 * (i - 1) + 1
    pred_idx = 2 * (i - 1) + 2

    symbolic_prey = sol_symbolic.u[end][prey_idx]
    flat_prey = sol_flat.u[end][prey_idx]
    symbolic_pred = sol_symbolic.u[end][pred_idx]
    flat_pred = sol_flat.u[end][pred_idx]

    prey_match = isapprox(symbolic_prey, flat_prey, rtol=1e-10)
    pred_match = isapprox(symbolic_pred, flat_pred, rtol=1e-10)

    if prey_match && pred_match
        println("  ✓ ODE $i: Both APIs match exactly")
    else
        println("  ✗ ODE $i: Mismatch detected!")
        all_match = false
    end
end

println()
if all_match
    println("✓ SUCCESS: Symbolic and flat vector APIs produce identical results!")
    println()
    println("Recommendation: Use the symbolic parameters API for better readability")
    println("and to avoid manual parameter ordering.")
else
    println("✗ ERROR: APIs do not match!")
end
println()
println("="^70)
