@testitem "Standard OrdinaryDiffEq.jl workflow" begin
    using LockstepODE
    using OrdinaryDiffEq: ODEProblem, Tsit5, solve

    # Simple harmonic oscillator: dx/dt = v, dv/dt = -x
    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    u0 = [1.0, 0.0, 2.0, 0.0]
    tspan = (0.0, 1.0)
    lockstep_func = LockstepFunction(harmonic_oscillator!, 2, 2)

    # Test standard workflow - manually batch u0
    prob = ODEProblem(lockstep_func, u0, tspan)
    sol = solve(prob, Tsit5())
    @test sol.t[1] ≈ 0.0
    @test sol.t[end] ≈ 1.0
    @test length(sol.u[1]) == 4  # Total state size

    # Extract individual solutions
    individual_sols = extract_solutions(lockstep_func, sol)
    @test length(individual_sols) == 2
    @test length(individual_sols[1].u) == length(sol.u)
    @test individual_sols[1].t == sol.t
    @test individual_sols[1].u[1] ≈ [1.0, 0.0]
    @test individual_sols[2].u[1] ≈ [2.0, 0.0]
end

@testitem "Array type dispatch" begin
    using LockstepODE
    import KernelAbstractions as KA

    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    u0 = [1.0, 0.0, 2.0, 0.0]

    # Test that regular arrays work with CPU implementation
    lockstep_func = LockstepFunction(harmonic_oscillator!, 2, 2)
    du = zeros(4)
    lockstep_func(du, u0, nothing, 0.0)
    @test du[1] ≈ 0.0
    @test du[2] ≈ -1.0

    # Test backend detection utility still works
    @test KA.get_backend(u0) isa KA.CPU
end
