using TestItemRunner: @testitem

@testitem "MTK Integration - Simple Decay" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    # Define a simple exponential decay system
    @parameters α
    @variables x(t)

    eqs = [D(x) ~ -α * x]
    @named decay_sys = ODESystem(eqs, t)

    # Create LockstepFunction with 10 instances
    num_odes = 10
    lockstep_func = LockstepFunction(decay_sys, 1, num_odes)

    # Setup initial conditions and parameters
    u0 = ones(num_odes)
    tspan = (0.0, 5.0)
    p = 0.5  # decay rate

    # Solve with tighter tolerances for accurate comparison
    prob = ODEProblem(lockstep_func, u0, tspan, p)
    sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10)

    # Verify solution - should match exp(-α*t)
    # At t=5.0, u should be exp(-0.5*5.0) = exp(-2.5) ≈ 0.0821
    expected = exp(-p * tspan[2])
    for i in 1:num_odes
        @test sol.u[end][i] ≈ expected rtol=1e-6
    end
end

@testitem "MTK Integration - Harmonic Oscillator" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    # Define harmonic oscillator: d²x/dt² = -ω²x
    # Convert to first-order system:
    # dx/dt = v
    # dv/dt = -ω²x
    @parameters ω
    @variables x(t) v(t)

    eqs = [
        D(x) ~ v,
        D(v) ~ -ω^2 * x
    ]
    @named oscillator_sys = ODESystem(eqs, t)

    # Create LockstepFunction with 5 instances
    num_odes = 5
    lockstep_func = LockstepFunction(oscillator_sys, 2, num_odes)

    # Setup initial conditions: x=1, v=0 (starting at maximum displacement)
    u0 = repeat([1.0, 0.0], num_odes)
    tspan = (0.0, 2π)
    p = 1.0  # ω = 1

    # Solve with tighter tolerances
    prob = ODEProblem(lockstep_func, u0, tspan, p)
    sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10)

    # Verify solution - should return to initial position after full period
    # At t=2π with ω=1, should be back at x=1, v=0
    for i in 1:num_odes
        x_idx = 2(i-1) + 1
        v_idx = 2(i-1) + 2
        @test sol.u[end][x_idx] ≈ 1.0 rtol=1e-5
        @test sol.u[end][v_idx] ≈ 0.0 atol=1e-5
    end
end

@testitem "MTK Integration - Per-ODE Parameters" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    # Define exponential growth system
    @parameters r
    @variables x(t)

    eqs = [D(x) ~ r * x]
    @named growth_sys = ODESystem(eqs, t)

    # Create LockstepFunction
    num_odes = 4
    lockstep_func = LockstepFunction(growth_sys, 1, num_odes)

    # Different growth rates for each ODE
    u0 = ones(num_odes)
    tspan = (0.0, 2.0)
    growth_rates = [0.1, 0.2, 0.3, 0.4]

    # Solve
    prob = ODEProblem(lockstep_func, u0, tspan, growth_rates)
    sol = solve(prob, Tsit5())

    # Verify each ODE has different growth
    for i in 1:num_odes
        expected = exp(growth_rates[i] * tspan[2])
        @test sol.u[end][i] ≈ expected rtol=1e-6
    end

    # Also verify they're all different
    final_values = sol.u[end]
    @test length(unique(final_values)) == num_odes
end

@testitem "MTK Integration - CPU Threading" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    # Define a system that's expensive enough to benefit from threading
    @parameters a b c
    @variables x(t) y(t) z(t)

    eqs = [
        D(x) ~ a * (y - x),
        D(y) ~ x * (b - z) - y,
        D(z) ~ x * y - c * z
    ]
    @named lorenz_sys = ODESystem(eqs, t)

    num_odes = 50

    # Test with threading enabled
    lockstep_threaded = LockstepFunction(lorenz_sys, 3, num_odes, internal_threading=true)
    u0 = repeat([1.0, 1.0, 1.0], num_odes)
    tspan = (0.0, 10.0)
    p = [10.0, 28.0, 8/3]

    prob_threaded = ODEProblem(lockstep_threaded, u0, tspan, p)
    sol_threaded = solve(prob_threaded, Tsit5())

    # Test without threading (for comparison)
    lockstep_serial = LockstepFunction(lorenz_sys, 3, num_odes, internal_threading=false)
    prob_serial = ODEProblem(lockstep_serial, u0, tspan, p)
    sol_serial = solve(prob_serial, Tsit5())

    # Both should give same results
    @test sol_threaded.u[end] ≈ sol_serial.u[end] rtol=1e-10
end

@testitem "MTK Integration - PerODE Memory Ordering" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    # Define simple 2D system
    @parameters α β
    @variables x(t) y(t)

    eqs = [
        D(x) ~ -α * x,
        D(y) ~ -β * y
    ]
    @named sys = ODESystem(eqs, t)

    num_odes = 3

    # Test PerODE ordering works with MTK
    lockstep_per_ode = LockstepFunction(sys, 2, num_odes, ordering=PerODE())
    u0_per_ode = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # [x1,y1, x2,y2, x3,y3]
    prob_per_ode = ODEProblem(lockstep_per_ode, u0_per_ode, (0.0, 1.0), [0.5, 0.3])
    sol_per_ode = solve(prob_per_ode, Tsit5(), abstol=1e-10, reltol=1e-10)

    # Just verify it runs and produces reasonable output
    @test length(sol_per_ode.u) > 1
    @test length(sol_per_ode.u[end]) == 6
    @test all(sol_per_ode.u[end] .> 0)  # All should decay but stay positive

    # Note: PerIndex ordering with MTK has variable ordering complexities
    # due to MTK's internal symbolic variable management. For now, we
    # recommend using PerODE ordering (the default) with MTK systems.
end

@testitem "MTK Integration - Single-threaded Fallback" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    # Simple system
    @parameters k
    @variables x(t)

    eqs = [D(x) ~ -k * x]
    @named sys = ODESystem(eqs, t)

    # Create with internal_threading=false
    lockstep_func = LockstepFunction(sys, 1, 5, internal_threading=false)

    u0 = ones(5)
    tspan = (0.0, 3.0)
    p = 0.4

    prob = ODEProblem(lockstep_func, u0, tspan, p)
    sol = solve(prob, Tsit5())

    # Verify correct solution
    expected = exp(-p * tspan[2])
    for i in 1:5
        @test sol.u[end][i] ≈ expected rtol=1e-6
    end
end

@testitem "MTK Integration - Multi-Parameter Per-ODE (Lotka-Volterra)" begin
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

    # Different parameter sets for each ODE
    # Format: Must match MTK's canonical parameter ordering (obtained via parameters())
    # MTK reorders parameters internally: [α, β, δ, γ] (NOT [α, β, γ, δ])
    sys_simplified = structural_simplify(lotka_volterra)
    param_order = parameters(sys_simplified)

    # Flatten parameters for LockstepODE according to MTK's ordering: [α, β, δ, γ]
    # Each ODE gets its own set of 4 parameters
    param_sets = [
        [1.5, 1.0, 1.0, 3.0],  # ODE 1: [α=1.5, β=1.0, δ=1.0, γ=3.0]
        [1.0, 2.0, 1.5, 2.0],  # ODE 2: [α=1.0, β=2.0, δ=1.5, γ=2.0]
        [1.0, 1.0, 2.0, 1.0]   # ODE 3: [α=1.0, β=1.0, δ=2.0, γ=1.0]
    ]
    params_flat = vcat(param_sets...)

    # Initial conditions: [prey, predator] for each ODE
    u0 = vcat([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
    tspan = (0.0, 10.0)

    # Solve batched system
    prob = ODEProblem(lockstep_func, u0, tspan, params_flat)
    sol = solve(prob, Tsit5())

    # Verify results by solving each ODE individually and comparing
    for i in 1:num_odes
        # Solve individual ODE with same parameters
        individual_prob = ODEProblem(sys_simplified, [1.0, 1.0], tspan, param_sets[i])
        individual_sol = solve(individual_prob, Tsit5())

        # Extract from batched solution
        prey_idx = 2 * (i - 1) + 1
        pred_idx = 2 * (i - 1) + 2
        batched_prey = sol.u[end][prey_idx]
        batched_pred = sol.u[end][pred_idx]

        individual_prey = individual_sol.u[end][1]
        individual_pred = individual_sol.u[end][2]

        # Check if they match - this verifies parameter ordering is preserved
        # Use rtol=1e-2 (1%) to account for numerical differences in solver trajectories
        # Lotka-Volterra is sensitive to numerical integration differences
        @test batched_prey ≈ individual_prey rtol=1e-2
        @test batched_pred ≈ individual_pred rtol=1e-2
    end

    # Verify that different parameter sets produce different results
    final_prey_values = [sol.u[end][2 * (i - 1) + 1] for i in 1:num_odes]
    @test length(unique(final_prey_values)) == num_odes  # All should be different
end

@testitem "MTK Integration - Symbolic Parameters (Dict)" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    # Define Lotka-Volterra system
    @parameters α β γ δ
    @variables x(t) y(t)

    eqs = [
        D(x) ~ α * x - β * x * y,
        D(y) ~ δ * x * y - γ * y
    ]
    @named lotka_volterra = ODESystem(eqs, t)

    # Create LockstepFunction
    num_odes = 3
    lockstep_func = LockstepFunction(lotka_volterra, 2, num_odes)

    # Specify parameters symbolically using Dict - order doesn't matter!
    params_symbolic = [
        Dict(α=>1.5, β=>1.0, γ=>3.0, δ=>1.0),
        Dict(α=>1.0, β=>2.0, γ=>2.0, δ=>1.5),
        Dict(α=>1.0, β=>1.0, γ=>1.0, δ=>2.0)
    ]

    u0 = vcat([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
    tspan = (0.0, 10.0)

    # Solve using symbolic parameters
    prob_symbolic = ODEProblem(lockstep_func, u0, tspan, params_symbolic)
    sol_symbolic = solve(prob_symbolic, Tsit5())

    # Compare against individual MTK solutions
    sys_simplified = structural_simplify(lotka_volterra)

    for i in 1:num_odes
        # Solve individual ODE
        individual_prob = ODEProblem(sys_simplified, [1.0, 1.0], tspan, params_symbolic[i])
        individual_sol = solve(individual_prob, Tsit5())

        # Extract from batched solution
        prey_idx = 2 * (i - 1) + 1
        pred_idx = 2 * (i - 1) + 2

        @test sol_symbolic.u[end][prey_idx] ≈ individual_sol.u[end][1] rtol=1e-2
        @test sol_symbolic.u[end][pred_idx] ≈ individual_sol.u[end][2] rtol=1e-2
    end

    # Verify different results for different parameters
    final_prey_values = [sol_symbolic.u[end][2 * (i - 1) + 1] for i in 1:num_odes]
    @test length(unique(final_prey_values)) == num_odes
end

@testitem "MTK Integration - Symbolic Parameters (Pairs)" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    # Define simple exponential growth
    @parameters r k
    @variables x(t)

    eqs = [D(x) ~ r * x * (1 - x/k)]  # Logistic growth
    @named logistic = ODESystem(eqs, t)

    num_odes = 4
    lockstep_func = LockstepFunction(logistic, 1, num_odes)

    # Specify parameters using Pair vectors
    params_pairs = [
        [r=>0.1, k=>10.0],
        [r=>0.2, k=>15.0],
        [r=>0.3, k=>20.0],
        [r=>0.4, k=>25.0]
    ]

    u0 = ones(num_odes) * 0.1  # Start at 10% of capacity
    tspan = (0.0, 50.0)

    prob = ODEProblem(lockstep_func, u0, tspan, params_pairs)
    sol = solve(prob, Tsit5())

    # Verify each approaches its carrying capacity
    sys_simplified = structural_simplify(logistic)

    for i in 1:num_odes
        # Solve individual ODE
        individual_prob = ODEProblem(sys_simplified, [0.1], tspan, Dict(params_pairs[i]))
        individual_sol = solve(individual_prob, Tsit5())

        @test sol.u[end][i] ≈ individual_sol.u[end][1] rtol=1e-2
    end
end
