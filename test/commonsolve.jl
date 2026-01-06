@testitem "solve via CommonSolve" begin
    using LockstepODE
    using OrdinaryDiffEq
    using SciMLBase

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))

    sol = solve(prob, Tsit5())

    @test sol isa LockstepSolution
    @test length(sol) == 3
    @test sol.retcode == ReturnCode.Success

    # Check each solution decays correctly
    for (i, s) in enumerate(sol.solutions)
        u0 = u0s[i][1]
        expected_final = u0 * exp(-1.0)
        @test isapprox(s.u[end][1], expected_final, rtol=1e-4)
    end
end

@testitem "solve with shared parameters" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    p = 0.5  # Shared parameter

    prob = LockstepProblem(lf, u0s, (0.0, 2.0), p)
    sol = solve(prob, Tsit5())

    for (i, s) in enumerate(sol.solutions)
        u0 = u0s[i][1]
        expected_final = u0 * exp(-0.5 * 2.0)
        @test isapprox(s.u[end][1], expected_final, rtol=1e-4)
    end
end

@testitem "solve with per-ODE parameters" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [1.0], [1.0]]
    ps = [0.5, 1.0, 2.0]  # Different decay rates

    prob = LockstepProblem(lf, u0s, (0.0, 1.0), ps)
    sol = solve(prob, Tsit5())

    for (i, s) in enumerate(sol.solutions)
        expected_final = exp(-ps[i] * 1.0)
        @test isapprox(s.u[end][1], expected_final, rtol=1e-4)
    end
end

@testitem "solve! from init" begin
    using LockstepODE
    using OrdinaryDiffEq
    using SciMLBase

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))

    integ = init(prob, Tsit5())
    sol = solve!(integ)

    @test sol isa LockstepSolution
    @test length(sol) == 3
    @test sol.retcode == ReturnCode.Success
end

@testitem "solve! after partial stepping" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))

    integ = init(prob, Tsit5())

    # Step partway
    LockstepODE.step!(integ, 0.3, true)
    @test integ.t ≈ 0.3 atol=1e-10

    # Complete the solve
    sol = solve!(integ)

    # Solution should go to t=1.0
    for s in sol.solutions
        @test s.t[end] ≈ 1.0 atol=1e-10
    end
end

@testitem "solve independent mode" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    sol = solve(prob, Tsit5())

    # Solutions should be independent
    for (i, s) in enumerate(sol.solutions)
        u0 = u0s[i][1]
        expected_final = u0 * exp(-1.0)
        @test isapprox(s.u[end][1], expected_final, rtol=1e-4)
    end
end

@testitem "harmonic oscillator accuracy" begin
    using LockstepODE
    using OrdinaryDiffEq

    function harmonic!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lf = LockstepFunction(harmonic!, 2, 2)

    # Two oscillators with different initial conditions
    u0s = [[1.0, 0.0], [0.0, 1.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 2π))
    sol = solve(prob, Tsit5(); abstol=1e-10, reltol=1e-10)

    # After one period, should return to initial conditions
    for (i, s) in enumerate(sol.solutions)
        @test isapprox(s.u[end][1], u0s[i][1], atol=1e-6)
        @test isapprox(s.u[end][2], u0s[i][2], atol=1e-6)
    end
end

@testitem "single u0 replication" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 5)

    # Single u0 should be replicated for all ODEs
    u0 = [1.0]
    prob = LockstepProblem(lf, u0, (0.0, 1.0))
    sol = solve(prob, Tsit5())

    expected_final = exp(-1.0)
    for s in sol.solutions
        @test isapprox(s.u[end][1], expected_final, rtol=1e-4)
    end
end

@testitem "solver kwargs passthrough" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))

    # Test that kwargs like abstol/reltol are passed through
    sol = solve(prob, Tsit5(); abstol=1e-12, reltol=1e-12)

    # Should have higher accuracy
    for (i, s) in enumerate(sol.solutions)
        u0 = u0s[i][1]
        expected_final = u0 * exp(-1.0)
        @test isapprox(s.u[end][1], expected_final, rtol=1e-10)
    end
end
