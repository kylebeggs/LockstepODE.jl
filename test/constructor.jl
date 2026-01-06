@testitem "LockstepFunction basic constructor" begin
    using LockstepODE

    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lf = LockstepFunction(harmonic_oscillator!, 2, 3)
    @test lf.num_odes == 3
    @test lf.ode_size == 2
    @test lf.callbacks === nothing
    @test lf.f === harmonic_oscillator!
end

@testitem "LockstepFunction validation" begin
    using LockstepODE

    function f!(du, u, p, t)
        du[1] = -u[1]
    end

    # Invalid num_odes
    @test_throws ArgumentError LockstepFunction(f!, 1, 0)
    @test_throws ArgumentError LockstepFunction(f!, 1, -1)

    # Invalid ode_size
    @test_throws ArgumentError LockstepFunction(f!, 0, 10)
    @test_throws ArgumentError LockstepFunction(f!, -1, 10)
end

@testitem "LockstepFunction with callbacks" begin
    using LockstepODE
    using OrdinaryDiffEq

    function f!(du, u, p, t)
        du[1] = u[1]  # exponential growth
    end

    # Single callback (shared)
    cb = DiscreteCallback((u, t, i) -> u[1] > 5.0, i -> (i.u[1] = 1.0))
    lf = LockstepFunction(f!, 1, 3; callbacks = cb)
    @test lf.callbacks === cb

    # Vector of callbacks (per-ODE)
    cbs = [
        DiscreteCallback((u, t, i) -> u[1] > 5.0, i -> (i.u[1] = 1.0)),
        DiscreteCallback((u, t, i) -> u[1] > 10.0, i -> (i.u[1] = 1.0)),
        DiscreteCallback((u, t, i) -> u[1] > 15.0, i -> (i.u[1] = 1.0))
    ]
    lf2 = LockstepFunction(f!, 1, 3; callbacks = cbs)
    @test lf2.callbacks === cbs
end
