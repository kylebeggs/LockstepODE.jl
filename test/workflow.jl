@testitem "Basic independent solve" begin
    using LockstepODE
    using OrdinaryDiffEq
    using SciMLBase

    # Exponential decay: du/dt = -u
    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)

    # Different initial conditions
    u0s = [[1.0], [2.0], [3.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

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

@testitem "Independent solve with shared parameters" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    p = 0.5  # Shared parameter

    sol = LockstepODE.solve(lf, u0s, (0.0, 2.0), p, Tsit5())

    for (i, s) in enumerate(sol.solutions)
        u0 = u0s[i][1]
        expected_final = u0 * exp(-0.5 * 2.0)
        @test isapprox(s.u[end][1], expected_final, rtol=1e-4)
    end
end

@testitem "Independent solve with per-ODE parameters" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [1.0], [1.0]]
    ps = [0.5, 1.0, 2.0]  # Different decay rates

    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), ps, Tsit5())

    for (i, s) in enumerate(sol.solutions)
        expected_final = exp(-ps[i] * 1.0)
        @test isapprox(s.u[end][1], expected_final, rtol=1e-4)
    end
end

@testitem "Solution accessors" begin
    using LockstepODE
    using OrdinaryDiffEq
    using SciMLBase

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

    # Test indexing
    @test sol[1] === sol.solutions[1]
    @test sol[2] === sol.solutions[2]
    @test sol[3] === sol.solutions[3]

    # Test length
    @test length(sol) == 3

    # Test iteration
    collected = collect(sol)
    @test length(collected) == 3
    for s in collected
        @test s isa ODESolution
    end

    # Test eachindex
    @test collect(eachindex(sol)) == [1, 2, 3]

    # Test firstindex/lastindex
    @test firstindex(sol) == 1
    @test lastindex(sol) == 3
end

@testitem "extract_solutions compatibility" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

    # extract_solutions should return the solutions vector
    extracted = extract_solutions(lf, sol)
    @test extracted === sol.solutions
    @test length(extracted) == 3
end

@testitem "extract_at_time" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

    # Extract at t=0.5
    states = extract_at_time(sol, 0.5)
    @test length(states) == 3

    for (i, state) in enumerate(states)
        u0 = u0s[i][1]
        expected = u0 * exp(-0.5)
        @test isapprox(state[1], expected, rtol=1e-4)
    end
end

@testitem "Harmonic oscillator accuracy" begin
    using LockstepODE
    using OrdinaryDiffEq

    function harmonic!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lf = LockstepFunction(harmonic!, 2, 2)

    # Two oscillators with different initial conditions
    u0s = [[1.0, 0.0], [0.0, 1.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 2π), Tsit5(); abstol=1e-10, reltol=1e-10)

    # After one period, should return to initial conditions
    for (i, s) in enumerate(sol.solutions)
        @test isapprox(s.u[end][1], u0s[i][1], atol=1e-6)
        @test isapprox(s.u[end][2], u0s[i][2], atol=1e-6)
    end
end

@testitem "Single u0 replication" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 5)

    # Single u0 should be replicated for all ODEs
    u0 = [1.0]
    sol = LockstepODE.solve(lf, u0, (0.0, 1.0), Tsit5())

    expected_final = exp(-1.0)
    for s in sol.solutions
        @test isapprox(s.u[end][1], expected_final, rtol=1e-4)
    end
end
