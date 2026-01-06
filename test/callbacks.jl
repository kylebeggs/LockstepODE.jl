@testitem "Per-ODE DiscreteCallback" begin
    using LockstepODE
    using OrdinaryDiffEq

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

    lf = LockstepFunction(growth!, 1, 2; callbacks = [cb1, cb2])
    u0s = [[1.0], [1.0]]
    ps = [1.0, 1.0]  # Same growth rate

    prob = LockstepProblem(lf, u0s, (0.0, 5.0), ps)
    sol = solve(prob, Tsit5())

    # Verify callbacks were hit (ODE 1 should reset more often)
    @test length(callback_hits[1]) > length(callback_hits[2])
    @test length(callback_hits[1]) > 0

    # Verify final states are within bounds
    @test sol[1].u[end][1] < 5.5
    @test sol[2].u[end][1] < 10.5
end

@testitem "Shared DiscreteCallback" begin
    using LockstepODE
    using OrdinaryDiffEq

    # Exponential decay
    function decay!(du, u, p, t)
        du[1] = -0.5 * u[1]
    end

    # Shared callback: prevent going below 0.1
    floor_cb = DiscreteCallback(
        (u, t, integrator) -> u[1] < 0.1,
        integrator -> (integrator.u[1] = 0.1)
    )

    lf = LockstepFunction(decay!, 1, 3; callbacks = floor_cb)
    u0s = [[2.0], [1.0], [0.5]]

    prob = LockstepProblem(lf, u0s, (0.0, 10.0))
    sol = solve(prob, Tsit5())

    # All should be floored at ~0.1
    for s in sol.solutions
        @test s.u[end][1] >= 0.09
    end
end

@testitem "ContinuousCallback bouncing ball" begin
    using LockstepODE
    using OrdinaryDiffEq

    # Falling ball with gravity
    function falling_ball!(du, u, p, t)
        du[1] = u[2]      # velocity
        du[2] = -9.8      # gravity
    end

    # Bounce when hitting ground
    bounce_cb = ContinuousCallback(
        (u, t, integrator) -> u[1],  # Event: position crosses zero
        integrator -> (integrator.u[2] = -0.8 * integrator.u[2])  # Reverse velocity with damping
    )

    lf = LockstepFunction(falling_ball!, 2, 2; callbacks = bounce_cb)
    u0s = [[2.0, 0.0], [3.0, 0.0]]

    prob = LockstepProblem(lf, u0s, (0.0, 3.0))
    sol = solve(prob, Tsit5())

    # Just verify solution exists
    @test length(sol) == 2
    @test length(sol[1].u) > 0
end

@testitem "No callbacks" begin
    using LockstepODE
    using OrdinaryDiffEq
    using SciMLBase

    function harmonic!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lf = LockstepFunction(harmonic!, 2, 2)

    u0s = [[1.0, 0.0], [2.0, 0.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    sol = solve(prob, Tsit5())

    @test sol.retcode == ReturnCode.Success
    @test length(sol) == 2
end
