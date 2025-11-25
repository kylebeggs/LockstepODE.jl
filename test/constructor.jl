@testitem "LockstepFunction basic constructor" begin
    using LockstepODE

    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lf = LockstepFunction(harmonic_oscillator!, 2, 3)
    @test lf.num_odes == 3
    @test lf.ode_size == 2
    @test lf.sync_interval == 0.0
    @test lf.coupling === nothing
    @test lf.coupling_indices === nothing
    @test lf.callbacks === nothing
    @test lf.f === harmonic_oscillator!
end

@testitem "LockstepFunction with sync_interval" begin
    using LockstepODE

    function f!(du, u, p, t)
        du[1] = -u[1]
    end

    lf = LockstepFunction(f!, 1, 10; sync_interval=0.5)
    @test lf.sync_interval == 0.5
end

@testitem "LockstepFunction with coupling" begin
    using LockstepODE

    function f!(du, u, p, t)
        du[1] = -u[1]
    end

    function couple!(states, t)
        mean_val = sum(s[1] for s in states) / length(states)
        for s in states
            s[1] = mean_val
        end
    end

    lf = LockstepFunction(f!, 1, 10;
        sync_interval=0.1,
        coupling=couple!,
        coupling_indices=[1]
    )
    @test lf.sync_interval == 0.1
    @test lf.coupling === couple!
    @test lf.coupling_indices == [1]
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

    # Invalid sync_interval
    @test_throws ArgumentError LockstepFunction(f!, 1, 10; sync_interval=-0.1)

    # Invalid coupling_indices
    @test_throws ArgumentError LockstepFunction(f!, 1, 10; coupling_indices=[0])
    @test_throws ArgumentError LockstepFunction(f!, 1, 10; coupling_indices=[2])  # out of bounds for ode_size=1
end

@testitem "LockstepFunction with callbacks" begin
    using LockstepODE
    using OrdinaryDiffEq

    function f!(du, u, p, t)
        du[1] = u[1]  # exponential growth
    end

    # Single callback (shared)
    cb = DiscreteCallback((u, t, i) -> u[1] > 5.0, i -> (i.u[1] = 1.0))
    lf = LockstepFunction(f!, 1, 3; callbacks=cb)
    @test lf.callbacks === cb

    # Vector of callbacks (per-ODE)
    cbs = [
        DiscreteCallback((u, t, i) -> u[1] > 5.0, i -> (i.u[1] = 1.0)),
        DiscreteCallback((u, t, i) -> u[1] > 10.0, i -> (i.u[1] = 1.0)),
        DiscreteCallback((u, t, i) -> u[1] > 15.0, i -> (i.u[1] = 1.0))
    ]
    lf2 = LockstepFunction(f!, 1, 3; callbacks=cbs)
    @test lf2.callbacks === cbs
end

@testitem "LockstepFunction warns on sync without coupling" begin
    using LockstepODE

    function f!(du, u, p, t)
        du[1] = -u[1]
    end

    # Should warn when sync_interval > 0 but no coupling provided
    @test_logs (:warn, r"sync_interval.*no coupling") LockstepFunction(f!, 1, 10; sync_interval=0.1)
end
