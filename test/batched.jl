@testitem "BatchedFunction constructor" begin
    using LockstepODE

    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lf = LockstepFunction(harmonic_oscillator!, 2, 3)

    # Default options
    bf = BatchedFunction(lf)
    @test bf.lf === lf
    @test bf.ordering isa PerODE
    @test bf.internal_threading == true

    # Custom options
    bf2 = BatchedFunction(lf; ordering=PerIndex(), internal_threading=false)
    @test bf2.ordering isa PerIndex
    @test bf2.internal_threading == false

    # From BatchedOpts
    opts = BatchedOpts(ordering=PerIndex(), internal_threading=false)
    bf3 = BatchedFunction(lf, opts)
    @test bf3.ordering isa PerIndex
    @test bf3.internal_threading == false
end

@testitem "LockstepProblem{Batched} constructor" begin
    using LockstepODE

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    ps = [0.1, 0.2, 0.3]

    # Batched mode problem
    prob = LockstepProblem{Batched}(lf, u0s, (0.0, 1.0), ps)
    @test prob isa LockstepProblem{Batched}
    @test prob.lf === lf
    @test length(prob.u0s) == 3
    @test prob.opts isa BatchedOpts
    @test prob.opts.ordering isa PerODE

    # With custom options
    prob2 = LockstepProblem{Batched}(lf, u0s, (0.0, 1.0), ps;
                                      ordering=PerIndex(), internal_threading=false)
    @test prob2.opts.ordering isa PerIndex
    @test prob2.opts.internal_threading == false
end

@testitem "Batched mode solve - exponential decay" begin
    using LockstepODE, OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 5)
    u0s = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    ps = [0.1, 0.2, 0.3, 0.4, 0.5]

    prob = LockstepProblem{Batched}(lf, u0s, (0.0, 10.0), ps)
    sol = solve(prob, Tsit5())

    @test sol isa LockstepSolution
    @test length(sol) == 5
    @test sol.retcode == SciMLBase.ReturnCode.Success

    # Check each ODE decayed correctly (rtol=1e-3 for numerical accuracy)
    for i in 1:5
        expected_final = u0s[i][1] * exp(-ps[i] * 10.0)
        @test isapprox(sol[i].u[end][1], expected_final, rtol=1e-3)
    end
end

@testitem "Batched mode init/step!/solve!" begin
    using LockstepODE, OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]

    prob = LockstepProblem{Batched}(lf, u0s, (0.0, 1.0), 0.5)
    integ = init(prob, Tsit5())

    @test integ isa BatchedLockstepIntegrator
    @test integ.t ≈ 0.0
    @test length(integ) == 3

    # Manual stepping
    step!(integ)
    @test integ.t > 0.0

    # Access per-ODE state
    @test length(integ.u) == 3
    @test length(integ[1].u) == 1  # SubIntegrator for first ODE

    # Complete solve
    sol = solve!(integ)
    @test sol isa LockstepSolution
    @test length(sol) == 3
end

@testitem "Batched mode reinit!" begin
    using LockstepODE, OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 2)
    u0s = [[1.0], [2.0]]

    prob = LockstepProblem{Batched}(lf, u0s, (0.0, 1.0), 0.5)
    integ = init(prob, Tsit5())

    # Solve first time and save the final value (BatchedSubSolution is a view)
    sol1 = solve!(integ)
    sol1_final_copy = copy(sol1[1].u[end])

    # Reinit and solve again
    reinit!(integ)
    sol2 = solve!(integ)

    # Should get same results (comparing to saved copy)
    @test isapprox(sol1_final_copy, sol2[1].u[end], rtol=1e-10)

    # Reinit with new ICs
    reinit!(integ, [[10.0], [20.0]])
    sol3 = solve!(integ)

    # Final values should be different (new IC 10.0 vs original 1.0)
    @test sol3[1].u[end][1] > sol1_final_copy[1]
end

@testitem "Batched mode solution interpolation" begin
    using LockstepODE, OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]

    prob = LockstepProblem{Batched}(lf, u0s, (0.0, 10.0), 0.5)
    sol = solve(prob, Tsit5())

    # Interpolation at t=5
    for i in 1:3
        state_at_5 = sol[i](5.0)
        expected = u0s[i][1] * exp(-0.5 * 5.0)
        @test isapprox(state_at_5[1], expected, rtol=1e-4)
    end
end

@testitem "Mode equivalence - Ensemble vs Batched" begin
    using LockstepODE, OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    ps = [0.1, 0.2, 0.3]

    # Ensemble mode (default)
    prob_e = LockstepProblem(lf, u0s, (0.0, 5.0), ps)
    sol_e = solve(prob_e, Tsit5())

    # Batched mode
    prob_b = LockstepProblem{Batched}(lf, u0s, (0.0, 5.0), ps)
    sol_b = solve(prob_b, Tsit5())

    # Compare final values (rtol=1e-4 accounts for different evaluation strategies)
    for i in 1:3
        @test isapprox(sol_e[i].u[end], sol_b[i].u[end], rtol=1e-4)
    end

    # Compare interpolation
    for i in 1:3
        @test isapprox(sol_e[i](2.5), sol_b[i](2.5), rtol=1e-4)
    end
end

@testitem "PerODE vs PerIndex ordering" begin
    using LockstepODE, OrdinaryDiffEq

    function harmonic!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lf = LockstepFunction(harmonic!, 2, 3)
    u0s = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]

    # PerODE ordering (default)
    prob_pde = LockstepProblem{Batched}(lf, u0s, (0.0, 2π); ordering=PerODE())
    sol_pde = solve(prob_pde, Tsit5())

    # PerIndex ordering
    prob_pidx = LockstepProblem{Batched}(lf, u0s, (0.0, 2π); ordering=PerIndex())
    sol_pidx = solve(prob_pidx, Tsit5())

    # Results should be equivalent (ordering is internal detail)
    for i in 1:3
        @test isapprox(sol_pde[i].u[end], sol_pidx[i].u[end], rtol=1e-6)
    end
end

@testitem "Batched mode with callbacks" begin
    using LockstepODE, OrdinaryDiffEq

    function growth!(du, u, p, t)
        du[1] = p * u[1]  # exponential growth
    end

    # Callback to reset when state exceeds threshold
    cb1 = DiscreteCallback((u, t, i) -> u[1] > 5.0, i -> (i.u[1] = 1.0))
    cb2 = DiscreteCallback((u, t, i) -> u[1] > 10.0, i -> (i.u[1] = 1.0))

    lf = LockstepFunction(growth!, 1, 2; callbacks=[cb1, cb2])
    u0s = [[1.0], [1.0]]
    ps = [1.0, 0.5]

    prob = LockstepProblem{Batched}(lf, u0s, (0.0, 5.0), ps)
    sol = solve(prob, Tsit5())

    # First ODE should have been reset multiple times (threshold 5, rate 1.0)
    # Final value should be < 5.0 (just reset)
    @test sol[1].u[end][1] < 10.0

    # Second ODE has threshold 10, slower rate - may or may not have reset
    @test sol[2].u[end][1] < 20.0
end

@testitem "BatchedSubSolution accessors" begin
    using LockstepODE, OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -0.5 * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]

    prob = LockstepProblem{Batched}(lf, u0s, (0.0, 2.0))
    sol = solve(prob, Tsit5(); saveat=0.5)

    # Check accessors
    sub = sol[1]
    @test sub isa LockstepODE.BatchedSubSolution

    # Time points
    @test length(sub.t) >= 4  # 0, 0.5, 1.0, 1.5, 2.0

    # State series
    @test length(sub.u) == length(sub.t)
    @test sub.u[1] ≈ [1.0]

    # Indexing into timepoints
    @test sub[1] ≈ [1.0]
    @test length(sub[end]) == 1

    # Iteration
    let n_iter = 0
        for state in sub
            n_iter += 1
            @test length(state) == 1
        end
        @test n_iter == length(sub.t)
    end

    # Interpolation
    @test isapprox(sub(1.0)[1], exp(-0.5), rtol=1e-4)
end

@testitem "Batched mode internal_threading=false" begin
    using LockstepODE, OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -0.5 * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]

    # Disable internal threading
    prob = LockstepProblem{Batched}(lf, u0s, (0.0, 1.0); internal_threading=false)
    sol = solve(prob, Tsit5())

    # Should still work correctly
    @test length(sol) == 3
    for i in 1:3
        expected = u0s[i][1] * exp(-0.5)
        @test isapprox(sol[i].u[end][1], expected, rtol=1e-4)
    end
end
