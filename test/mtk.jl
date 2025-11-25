@testitem "MTK simple decay" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    @parameters α
    @variables x(t)

    eqs = [D(x) ~ -α * x]
    @named decay_sys = ODESystem(eqs, t)

    lf = LockstepFunction(decay_sys, 3)

    u0s = [[1.0], [2.0], [3.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 2.0), 0.5, Tsit5())  # α = 0.5

    @test sol isa LockstepSolution
    @test length(sol) == 3

    # Verify each solution
    for (i, s) in enumerate(sol.solutions)
        expected = u0s[i][1] * exp(-0.5 * 2.0)
        @test isapprox(s.u[end][1], expected, rtol=1e-4)
    end
end

@testitem "MTK harmonic oscillator" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    @parameters ω
    @variables x(t) v(t)

    eqs = [
        D(x) ~ v,
        D(v) ~ -ω^2 * x
    ]
    @named oscillator = ODESystem(eqs, t)

    lf = LockstepFunction(oscillator, 3)

    u0s = [[1.0, 0.0], [2.0, 0.0], [0.5, 0.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 2π), 1.0, Tsit5(); abstol=1e-10, reltol=1e-10)  # ω = 1

    # After one period, should return to initial
    for (i, s) in enumerate(sol.solutions)
        @test isapprox(s.u[end][1], u0s[i][1], atol=1e-5)
        @test isapprox(s.u[end][2], u0s[i][2], atol=1e-5)
    end
end

@testitem "MTK with per-ODE parameters" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    @parameters r
    @variables x(t)

    eqs = [D(x) ~ r * x]
    @named growth = ODESystem(eqs, t)

    lf = LockstepFunction(growth, 3)

    u0s = [[1.0], [1.0], [1.0]]
    ps = [0.1, 0.5, 1.0]  # Different growth rates

    sol = LockstepODE.solve(lf, u0s, (0.0, 2.0), ps, Tsit5())

    for (i, s) in enumerate(sol.solutions)
        expected = exp(ps[i] * 2.0)
        @test isapprox(s.u[end][1], expected, rtol=1e-4)
    end

    # Verify they're all different
    finals = [s.u[end][1] for s in sol.solutions]
    @test length(unique(finals)) == 3
end

@testitem "MTK with callbacks" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    @parameters r
    @variables x(t)

    eqs = [D(x) ~ r * x]
    @named growth = ODESystem(eqs, t)

    reset_count = Ref(0)

    cb = DiscreteCallback(
        (u, t, integrator) -> u[1] > 5.0,
        integrator -> begin
            reset_count[] += 1
            integrator.u[1] = 1.0
        end
    )

    lf = LockstepFunction(growth, 3; callbacks=cb)

    u0s = [[1.0], [1.0], [1.0]]
    ps = [1.0, 1.0, 1.0]

    sol = LockstepODE.solve(lf, u0s, (0.0, 10.0), ps, Tsit5())

    @test reset_count[] > 0
    for s in sol.solutions
        @test s.u[end][1] < 5.5
    end
end

@testitem "MTK with coupling" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D
    using Statistics

    @parameters α
    @variables x(t)

    eqs = [D(x) ~ -α * x]
    @named decay = ODESystem(eqs, t)

    function couple!(states, t)
        m = mean(s[1] for s in states)
        for s in states
            s[1] = m
        end
    end

    lf = LockstepFunction(decay, 3;
        sync_interval=0.1,
        coupling=couple!
    )

    u0s = [[1.0], [2.0], [3.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), 0.5, Tsit5())

    # Should synchronize
    finals = [s.u[end][1] for s in sol.solutions]
    @test maximum(finals) - minimum(finals) < 0.1
end

@testitem "MTK transform_parameters" begin
    using LockstepODE
    using OrdinaryDiffEq
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D

    @parameters α β
    @variables x(t)

    eqs = [D(x) ~ -α * x + β]
    @named sys = ODESystem(eqs, t)
    # Use mtkcompile if available (MTK v10+), otherwise structural_simplify
    simplify_fn = isdefined(ModelingToolkit, :mtkcompile) ? ModelingToolkit.mtkcompile : ModelingToolkit.structural_simplify
    sys_simple = simplify_fn(sys)

    params = [
        Dict(α => 1.0, β => 0.5),
        Dict(α => 2.0, β => 1.0),
    ]

    # Get the extension module and use transform_parameters
    ext = Base.get_extension(LockstepODE, :LockstepODEMTKExt)
    transformed = ext.transform_parameters(sys_simple, params)

    @test length(transformed) == 2
    @test transformed[1] isa Vector{Float64}
    @test transformed[2] isa Vector{Float64}
end
