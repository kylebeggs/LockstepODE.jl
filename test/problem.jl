@testitem "LockstepProblem construction" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    tspan = (0.0, 1.0)

    prob = LockstepProblem(lf, u0s, tspan)

    @test prob.lf === lf
    @test prob.u0s == u0s
    @test prob.tspan == tspan
    @test length(prob.ps) == 3
    @test all(p === nothing for p in prob.ps)
end

@testitem "LockstepProblem with parameters" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    tspan = (0.0, 1.0)

    # Shared parameter (tuple)
    prob = LockstepProblem(lf, u0s, tspan, (0.5, 0.1))
    @test all(p == (0.5, 0.1) for p in prob.ps)

    # Per-ODE scalar parameters
    prob = LockstepProblem(lf, u0s, tspan, [0.5, 1.0, 2.0])
    @test prob.ps == [0.5, 1.0, 2.0]

    # Per-ODE vector parameters
    prob = LockstepProblem(lf, u0s, tspan, [[1, 2], [3, 4], [5, 6]])
    @test prob.ps == [[1, 2], [3, 4], [5, 6]]
end

@testitem "LockstepProblem single u0 replication" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 5)
    tspan = (0.0, 1.0)

    # Single u0 should be replicated
    prob = LockstepProblem(lf, [1.0], tspan)
    @test length(prob.u0s) == 5
    @test all(u0 == [1.0] for u0 in prob.u0s)
end

@testitem "LockstepProblem validation" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    tspan = (0.0, 1.0)

    # Wrong number of u0s
    @test_throws ArgumentError LockstepProblem(lf, [[1.0], [2.0]], tspan)

    # Wrong u0 size
    @test_throws ArgumentError LockstepProblem(lf, [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], tspan)
end

