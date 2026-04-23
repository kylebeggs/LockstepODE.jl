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

@testitem "LockstepProblem pre-flattened u0 (Batched, PerODE)" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
        du[2] = -p * u[2]
    end

    num_odes = 4
    ode_size = 2
    lf = LockstepFunction(decay!, ode_size, num_odes)
    ps = [0.1, 0.2, 0.3, 0.4]
    tspan = (0.0, 1.0)

    u0s_vov = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    # PerODE layout: [u0s_vov[1]..., u0s_vov[2]..., ...]
    u0_flat = vcat(u0s_vov...)

    prob_vov = LockstepProblem{Batched}(lf, u0s_vov, tspan, ps; ordering=PerODE())
    prob_flat = LockstepProblem{Batched}(lf, u0_flat, tspan, ps; ordering=PerODE())

    # Flat storage roundtrips the input array
    @test prob_flat.u0s === u0_flat
    @test prob_flat.u0s isa AbstractVector{<:Number}
    @test length(prob_flat.u0s) == num_odes * ode_size

    sol_vov = solve(prob_vov, Tsit5())
    sol_flat = solve(prob_flat, Tsit5())

    for i in 1:num_odes
        @test isapprox(sol_vov[i].u[end], sol_flat[i].u[end]; rtol=1e-8)
    end
end

@testitem "LockstepProblem pre-flattened u0 (Batched, PerIndex)" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -p * u[1]
        du[2] = -p * u[2]
    end

    num_odes = 3
    ode_size = 2
    lf = LockstepFunction(decay!, ode_size, num_odes)
    ps = [0.5, 1.0, 2.0]
    tspan = (0.0, 1.0)

    u0s_vov = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    # PerIndex layout: [u1_ode1, u1_ode2, u1_ode3, u2_ode1, u2_ode2, u2_ode3]
    u0_flat = Float64[
        u0s_vov[1][1], u0s_vov[2][1], u0s_vov[3][1],
        u0s_vov[1][2], u0s_vov[2][2], u0s_vov[3][2],
    ]

    prob_vov = LockstepProblem{Batched}(lf, u0s_vov, tspan, ps; ordering=PerIndex())
    prob_flat = LockstepProblem{Batched}(lf, u0_flat, tspan, ps; ordering=PerIndex())

    sol_vov = solve(prob_vov, Tsit5())
    sol_flat = solve(prob_flat, Tsit5())

    for i in 1:num_odes
        @test isapprox(sol_vov[i].u[end], sol_flat[i].u[end]; rtol=1e-8)
    end
end

@testitem "LockstepProblem pre-flattened u0 length validation" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(decay!, 1, 4)  # ode_size=1, num_odes=4; valid lengths: 1 or 4
    tspan = (0.0, 1.0)

    # length=2 is neither ode_size (1) nor num_odes*ode_size (4)
    @test_throws ArgumentError LockstepProblem{Batched}(lf, [0.0, 1.0], tspan)
    # length=3 likewise invalid
    @test_throws ArgumentError LockstepProblem{Batched}(lf, [0.0, 1.0, 2.0], tspan)

    # Boundary cases that must succeed
    @test LockstepProblem{Batched}(lf, [1.0], tspan) isa LockstepProblem         # replicate
    @test LockstepProblem{Batched}(lf, [1.0, 2.0, 3.0, 4.0], tspan) isa LockstepProblem  # flat
end

@testitem "LockstepProblem flat u0 rejected in Ensemble mode" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
        du[2] = -u[2]
    end

    lf = LockstepFunction(decay!, 2, 3)  # ode_size=2, num_odes=3 → flat length 6
    tspan = (0.0, 1.0)

    u0_flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    @test_throws ArgumentError LockstepProblem{Ensemble}(lf, u0_flat, tspan)

    # Single-state replication (length=ode_size) still works in Ensemble mode
    @test LockstepProblem{Ensemble}(lf, [1.0, 2.0], tspan) isa LockstepProblem
end

