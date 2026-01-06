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
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    sol = solve(prob, Tsit5())

    @test sol isa LockstepSolution
    @test length(sol) == 3
    @test sol.retcode == ReturnCode.Success

    # Check each solution decays correctly
    for (i, s) in enumerate(sol.solutions)
        u0 = u0s[i][1]
        expected_final = u0 * exp(-1.0)
        @test isapprox(s.u[end][1], expected_final, rtol = 1e-4)
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

    prob = LockstepProblem(lf, u0s, (0.0, 2.0), p)
    sol = solve(prob, Tsit5())

    for (i, s) in enumerate(sol.solutions)
        u0 = u0s[i][1]
        expected_final = u0 * exp(-0.5 * 2.0)
        @test isapprox(s.u[end][1], expected_final, rtol = 1e-4)
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

    prob = LockstepProblem(lf, u0s, (0.0, 1.0), ps)
    sol = solve(prob, Tsit5())

    for (i, s) in enumerate(sol.solutions)
        expected_final = exp(-ps[i] * 1.0)
        @test isapprox(s.u[end][1], expected_final, rtol = 1e-4)
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

    # Test with Batched mode (default)
    prob_b = LockstepProblem(lf, u0s, (0.0, 1.0))
    sol_b = solve(prob_b, Tsit5())

    # Test indexing - Batched mode creates wrappers, check values match
    @test sol_b[1].u[end] ≈ sol_b.solutions[1].u[end]
    @test sol_b[2].u[end] ≈ sol_b.solutions[2].u[end]
    @test sol_b[3].u[end] ≈ sol_b.solutions[3].u[end]

    # Test with Ensemble mode (identity preserved)
    prob_e = LockstepProblem{Ensemble}(lf, u0s, (0.0, 1.0))
    sol_e = solve(prob_e, Tsit5())

    @test sol_e[1] === sol_e.solutions[1]
    @test sol_e[2] === sol_e.solutions[2]
    @test sol_e[3] === sol_e.solutions[3]

    # Test length (both modes)
    @test length(sol_b) == 3
    @test length(sol_e) == 3

    # Test iteration
    collected_b = collect(sol_b)
    collected_e = collect(sol_e)
    @test length(collected_b) == 3
    @test length(collected_e) == 3
    for s in collected_e
        @test s isa ODESolution
    end

    # Test eachindex
    @test collect(eachindex(sol_b)) == [1, 2, 3]
    @test collect(eachindex(sol_e)) == [1, 2, 3]

    # Test firstindex/lastindex
    @test firstindex(sol_b) == 1
    @test lastindex(sol_b) == 3
    @test firstindex(sol_e) == 1
    @test lastindex(sol_e) == 3
end

@testitem "extract_solutions compatibility" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    sol = solve(prob, Tsit5())

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
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    sol = solve(prob, Tsit5())

    # Extract at t=0.5
    states = extract_at_time(sol, 0.5)
    @test length(states) == 3

    for (i, state) in enumerate(states)
        u0 = u0s[i][1]
        expected = u0 * exp(-0.5)
        @test isapprox(state[1], expected, rtol = 1e-4)
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
    prob = LockstepProblem(lf, u0s, (0.0, 2π))
    sol = solve(prob, Tsit5(); abstol = 1e-10, reltol = 1e-10)

    # After one period, should return to initial conditions
    for (i, s) in enumerate(sol.solutions)
        @test isapprox(s.u[end][1], u0s[i][1], atol = 1e-6)
        @test isapprox(s.u[end][2], u0s[i][2], atol = 1e-6)
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
    prob = LockstepProblem(lf, u0, (0.0, 1.0))
    sol = solve(prob, Tsit5())

    expected_final = exp(-1.0)
    for s in sol.solutions
        @test isapprox(s.u[end][1], expected_final, rtol = 1e-4)
    end
end
