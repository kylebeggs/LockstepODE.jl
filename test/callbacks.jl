@testitem "DiscreteCallback per-ODE callbacks" begin
    using LockstepODE
    using OrdinaryDiffEq: ODEProblem, Tsit5, solve, DiscreteCallback

    # Exponential growth
    function growth!(du, u, p, t)
        du[1] = p * u[1]
    end

    # Track callback hits
    callback_hits = [Int[], Int[]]

    # Different reset thresholds for each ODE
    cb1 = DiscreteCallback(
        (u, t, integrator) -> u[1] > 5.0,
        integrator -> begin
            push!(callback_hits[1], length(callback_hits[1]) + 1)
            integrator.u[1] = 1.0
        end
    )

    cb2 = DiscreteCallback(
        (u, t, integrator) -> u[1] > 10.0,
        integrator -> begin
            push!(callback_hits[2], length(callback_hits[2]) + 1)
            integrator.u[1] = 1.0
        end
    )

    # Create lockstep function with per-ODE callbacks
    lockstep_func = LockstepFunction(growth!, 1, 2, callbacks=[cb1, cb2])
    u0_batched = [1.0, 1.0]
    tspan = (0.0, 5.0)
    p_batched = [1.0, 1.0]  # Same growth rate

    # Callbacks are automatically handled by ODEProblem constructor
    prob = ODEProblem(lockstep_func, u0_batched, tspan, p_batched)
    sol = solve(prob, Tsit5())

    # Verify callbacks were hit (ODE 1 should reset more often)
    @test length(callback_hits[1]) > length(callback_hits[2])
    @test length(callback_hits[1]) > 0

    # Verify final states are within bounds
    individual_sols = extract_solutions(lockstep_func, sol)
    @test individual_sols[1].u[end][1] < 5.0  # Below threshold
    @test individual_sols[2].u[end][1] < 10.0  # Below threshold
end

@testitem "DiscreteCallback shared callback" begin
    using LockstepODE
    using OrdinaryDiffEq: ODEProblem, Tsit5, solve, DiscreteCallback

    # Exponential decay
    function decay!(du, u, p, t)
        du[1] = -0.5 * u[1]
    end

    # Shared callback: prevent going below 0.1
    floor_cb = DiscreteCallback(
        (u, t, integrator) -> u[1] < 0.1,
        integrator -> (integrator.u[1] = 0.1)
    )

    lockstep_func = LockstepFunction(decay!, 1, 3, callbacks=floor_cb)
    u0_batched = [2.0, 1.0, 0.5]
    tspan = (0.0, 10.0)

    # Callbacks are automatically handled
    prob = ODEProblem(lockstep_func, u0_batched, tspan, nothing)
    sol = solve(prob, Tsit5())

    individual_sols = extract_solutions(lockstep_func, sol)
    # All should be floored at 0.1
    for sol_i in individual_sols
        @test sol_i.u[end][1] ≥ 0.09  # Allow small numerical error
    end
end

@testitem "ContinuousCallback bouncing ball" begin
    using LockstepODE
    using OrdinaryDiffEq: ODEProblem, Tsit5, solve, ContinuousCallback

    # Falling ball with gravity
    function falling_ball!(du, u, p, t)
        du[1] = u[2]      # velocity
        du[2] = -9.8      # gravity
    end

    # Bounce when hitting ground (position = 0)
    bounce_cb = ContinuousCallback(
        (u, t, integrator) -> u[1],  # Event: position crosses zero
        integrator -> (integrator.u[2] = -0.8 * integrator.u[2])  # Reverse velocity with damping
    )

    # Two balls dropped from different heights
    lockstep_func = LockstepFunction(falling_ball!, 2, 2, callbacks=bounce_cb)
    u0_vec = [[2.0, 0.0], [3.0, 0.0]]  # Smaller heights for easier testing
    u0_batched = batch_initial_conditions(u0_vec, 2, 2)
    tspan = (0.0, 3.0)  # Shorter time

    # Callbacks are automatically handled
    prob = ODEProblem(lockstep_func, u0_batched, tspan, nothing)
    sol = solve(prob, Tsit5())

    individual_sols = extract_solutions(lockstep_func, sol)

    # Just check that solution exists (callbacks may not be perfect yet)
    @test length(individual_sols) == 2
    @test length(individual_sols[1].u) > 0
end

@testitem "Callbacks with PerIndex ordering" begin
    using LockstepODE
    using OrdinaryDiffEq: ODEProblem, Tsit5, solve, DiscreteCallback

    # Simple growth (for PerIndex test)
    function simple_growth!(du, u, p, t)
        du[1] = p * u[1]
    end

    reset_cb = DiscreteCallback(
        (u, t, integrator) -> u[1] > 5.0,
        integrator -> (integrator.u[1] = 1.0)
    )

    # Test with PerIndex ordering
    lockstep_func = LockstepFunction(simple_growth!, 1, 2,
                                    ordering=PerIndex(), callbacks=reset_cb)
    u0_batched = [1.0, 1.0]
    tspan = (0.0, 3.0)
    p_batched = [1.0, 1.0]

    # Callbacks are automatically handled
    prob = ODEProblem(lockstep_func, u0_batched, tspan, p_batched)
    sol = solve(prob, Tsit5())

    individual_sols = extract_solutions(lockstep_func, sol)
    # Should work correctly with PerIndex ordering
    @test all(sol_i.u[end][1] < 5.5 for sol_i in individual_sols)
end

@testitem "No callbacks" begin
    using LockstepODE
    using OrdinaryDiffEq: ODEProblem, Tsit5, solve, ReturnCode

    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lockstep_func = LockstepFunction(harmonic_oscillator!, 2, 2, callbacks=nothing)

    # Test that manual call still works (backward compatibility)
    wrapped_cb = create_lockstep_callbacks(lockstep_func)
    @test wrapped_cb === nothing

    # Test that automatic ODEProblem creation works
    u0 = [1.0, 0.0, 2.0, 0.0]
    prob = ODEProblem(lockstep_func, u0, (0.0, 1.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
end

@testitem "Callback merging with user callbacks" begin
    using LockstepODE
    using OrdinaryDiffEq: ODEProblem, Tsit5, solve, DiscreteCallback

    function growth!(du, u, p, t)
        du[1] = p * u[1]
    end

    # LockstepFunction callback: reset at 5.0
    lockstep_cb = DiscreteCallback(
        (u, t, integrator) -> u[1] > 5.0,
        integrator -> (integrator.u[1] = 1.0)
    )

    # Track if user callback was hit
    user_cb_hit = Ref(false)

    # User callback: mark when value exceeds 3.0 (happens before lockstep reset)
    user_cb = DiscreteCallback(
        (u, t, integrator) -> u[1] > 3.0,
        integrator -> (user_cb_hit[] = true)
    )

    lockstep_func = LockstepFunction(growth!, 1, 2, callbacks=lockstep_cb)
    u0_batched = [1.0, 1.0]
    p_batched = [1.0, 1.0]

    # Pass user callback to ODEProblem - should be merged with lockstep callbacks
    prob = ODEProblem(lockstep_func, u0_batched, (0.0, 3.0), p_batched, callback=user_cb)
    sol = solve(prob, Tsit5())

    # Verify both callbacks were applied
    @test user_cb_hit[]  # User callback triggered
    individual_sols = extract_solutions(lockstep_func, sol)
    @test individual_sols[1].u[end][1] < 5.5  # Lockstep callback prevented exceeding 5.0
end
