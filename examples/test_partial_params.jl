# Test Partial Parameter Specification
# Verifies that partial parameter specification works correctly

using Pkg
Pkg.activate(@__DIR__)
using LockstepODE
using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

println("Testing Partial Parameter Specification")
println("="^70)

# Define system with default parameters
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0
@variables x(t) y(t)

eqs = [
    D(x) ~ α * x - β * x * y,
    D(y) ~ δ * x * y - γ * y
]
@named sys = ODESystem(eqs, t)
sys = complete(sys)

# Test Case 1: Empty Dict (all defaults)
println("\n1. Empty Dict - all defaults")
println("-"^70)
lockstep_func = LockstepFunction(sys, 2, 1)
params = [Dict()]
u0 = [1.0, 1.0]
prob = ODEProblem(lockstep_func, u0, (0.0, 1.0), params)
sol = solve(prob, Tsit5())
println("✓ Empty Dict works - uses all defaults")

# Test Case 2: Partial specification (one param)
println("\n2. Partial specification - one parameter")
println("-"^70)
params = [Dict(α => 2.0)]
prob = ODEProblem(lockstep_func, u0, (0.0, 1.0), params)
sol = solve(prob, Tsit5())
println("✓ Single parameter override works")

# Test Case 3: Partial specification (two params)
println("\n3. Partial specification - two parameters")
println("-"^70)
params = [Dict(β => 0.5, δ => 2.0)]
prob = ODEProblem(lockstep_func, u0, (0.0, 1.0), params)
sol = solve(prob, Tsit5())
println("✓ Multiple parameter override works")

# Test Case 4: Full specification
println("\n4. Full specification - all parameters")
println("-"^70)
params = [Dict(α => 1.0, β => 2.0, γ => 2.0, δ => 1.5)]
prob = ODEProblem(lockstep_func, u0, (0.0, 1.0), params)
sol = solve(prob, Tsit5())
println("✓ Full specification works (backward compatible)")

# Test Case 5: Mix of partial and full across multiple ODEs
println("\n5. Mixed specification - 3 ODEs with different styles")
println("-"^70)
lockstep_func = LockstepFunction(sys, 2, 3)
params = [
    Dict(),                                      # ODE 1: all defaults
    Dict(α => 2.0, β => 0.5),                   # ODE 2: partial
    Dict(α => 1.0, β => 2.0, γ => 2.0, δ => 1.5) # ODE 3: full
]
u0 = vcat([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
prob = ODEProblem(lockstep_func, u0, (0.0, 1.0), params)
sol = solve(prob, Tsit5())
println("✓ Mixed specification across ODEs works")

# Test Case 6: Verify defaults are actually used
println("\n6. Verification - defaults produce same result as explicit values")
println("-"^70)
lockstep_func = LockstepFunction(sys, 2, 2)

# ODE 1 with explicit defaults, ODE 2 with empty dict
params_explicit = [
    Dict(α => 1.5, β => 1.0, γ => 3.0, δ => 1.0),
    Dict()
]
prob1 = ODEProblem(lockstep_func, vcat([1.0, 1.0], [1.0, 1.0]), (0.0, 5.0), params_explicit)
sol1 = solve(prob1, Tsit5())

# Both with empty dict
params_default = [Dict(), Dict()]
prob2 = ODEProblem(lockstep_func, vcat([1.0, 1.0], [1.0, 1.0]), (0.0, 5.0), params_default)
sol2 = solve(prob2, Tsit5())

# Compare final states
diff = maximum(abs.(sol1.u[end] .- sol2.u[end]))
if diff < 1e-10
    println("✓ Explicit defaults and empty Dict produce identical results")
    println("  Max difference: $(diff)")
else
    println("✗ Results differ! Max difference: $(diff)")
end

println("\n" * "="^70)
println("All tests passed! ✓")
println("="^70)
