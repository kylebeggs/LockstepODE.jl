@testitem "LockstepIntegrator creation via init" begin
    using LockstepODE
    using OrdinaryDiffEq
    using SciMLBase

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]

    # Test both modes
    prob_batched = LockstepProblem(lf, u0s, (0.0, 1.0))
    prob_ensemble = LockstepProblem{Ensemble}(lf, u0s, (0.0, 1.0))

    integ_b = init(prob_batched, Tsit5())
    integ_e = init(prob_ensemble, Tsit5())

    @test integ_b isa LockstepIntegrator
    @test integ_e isa LockstepIntegrator
    @test length(integ_b) == 3
    @test length(integ_e) == 3
    @test integ_b.t == 0.0
    @test integ_e.t == 0.0
    @test integ_b.tspan == (0.0, 1.0)
    @test integ_e.tspan == (0.0, 1.0)
end

@testitem "LockstepIntegrator indexing" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    integ = init(prob, Tsit5())

    # Test indexing returns underlying integrators
    @test integ[1].u[1] ≈ 1.0
    @test integ[2].u[1] ≈ 2.0
    @test integ[3].u[1] ≈ 3.0

    # Test firstindex/lastindex
    @test firstindex(integ) == 1
    @test lastindex(integ) == 3

    # Test eachindex
    @test collect(eachindex(integ)) == [1, 2, 3]
end

@testitem "LockstepIntegrator property accessors" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    ps = [0.5, 1.0, 2.0]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0), ps)
    integ = init(prob, Tsit5())

    # Test .u returns vector of states
    @test integ.u isa Vector
    @test length(integ.u) == 3
    @test integ.u[1][1] ≈ 1.0
    @test integ.u[2][1] ≈ 2.0
    @test integ.u[3][1] ≈ 3.0

    # Test .p returns vector of parameters
    @test integ.p isa Vector
    @test length(integ.p) == 3
    @test integ.p == [0.5, 1.0, 2.0]

    # Test .t
    @test integ.t == 0.0
end

@testitem "LockstepIntegrator iteration" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]

    # Test Ensemble mode iteration (identity preserved)
    prob_e = LockstepProblem{Ensemble}(lf, u0s, (0.0, 1.0))
    integ_e = init(prob_e, Tsit5())

    collected_e = collect(integ_e)
    @test length(collected_e) == 3
    @test collected_e[1] === integ_e[1]
    @test collected_e[2] === integ_e[2]
    @test collected_e[3] === integ_e[3]

    # Test Batched mode iteration (values match, not identity since SubIntegrator is created on access)
    prob_b = LockstepProblem(lf, u0s, (0.0, 1.0))
    integ_b = init(prob_b, Tsit5())

    collected_b = collect(integ_b)
    @test length(collected_b) == 3
    @test collected_b[1].u ≈ integ_b[1].u
    @test collected_b[2].u ≈ integ_b[2].u
    @test collected_b[3].u ≈ integ_b[3].u
end

@testitem "step! single step" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    integ = init(prob, Tsit5())

    # Take a step
    LockstepODE.step!(integ)

    # Time should have advanced
    @test integ.t > 0.0

    # States should have changed
    @test integ.u[1][1] < 1.0
    @test integ.u[2][1] < 2.0
    @test integ.u[3][1] < 3.0
end

@testitem "step! with dt" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    integ = init(prob, Tsit5())

    # Step exactly to t=0.1
    LockstepODE.step!(integ, 0.1, true)

    # Should be at t=0.1
    @test integ.t ≈ 0.1 atol=1e-10

    # Check solution accuracy
    for (i, u0) in enumerate([1.0, 2.0, 3.0])
        expected = u0 * exp(-0.1)
        @test isapprox(integ[i].u[1], expected, rtol = 1e-4)
    end
end

@testitem "reinit!" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    integ = init(prob, Tsit5())

    # Step forward
    LockstepODE.step!(integ, 0.5, true)
    @test integ.t ≈ 0.5

    # Reinit
    LockstepODE.reinit!(integ)

    # Should be back at t=0
    @test integ.t == 0.0
    @test integ[1].u[1] ≈ 1.0
    @test integ[2].u[1] ≈ 2.0
    @test integ[3].u[1] ≈ 3.0
end

@testitem "reinit! with new u0s" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 3)
    u0s = [[1.0], [2.0], [3.0]]
    prob = LockstepProblem(lf, u0s, (0.0, 1.0))
    integ = init(prob, Tsit5())

    # Step forward
    LockstepODE.step!(integ, 0.5, true)

    # Reinit with new initial conditions
    new_u0s = [[5.0], [6.0], [7.0]]
    LockstepODE.reinit!(integ, new_u0s)

    # Should have new initial conditions
    @test integ.t == 0.0
    @test integ[1].u[1] ≈ 5.0
    @test integ[2].u[1] ≈ 6.0
    @test integ[3].u[1] ≈ 7.0
end
