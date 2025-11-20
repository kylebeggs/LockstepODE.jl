@testitem "LockstepFunction convenient constructor" begin
    using LockstepODE

    # Simple harmonic oscillator: dx/dt = v, dv/dt = -x
    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lockstep_func = LockstepFunction(harmonic_oscillator!, 2, 2)
    @test lockstep_func.num_odes == 2
    @test lockstep_func.ode_size == 2
    @test lockstep_func.internal_threading == true
    @test lockstep_func.ordering isa PerODE
    @test lockstep_func.wrapper isa SimpleWrapper
    @test lockstep_func.wrapper.f === harmonic_oscillator!
end

@testitem "LockstepFunction with options" begin
    using LockstepODE

    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lockstep_func = LockstepFunction(harmonic_oscillator!, 2, 2; internal_threading=false, ordering=PerIndex())
    @test lockstep_func.internal_threading == false
    @test lockstep_func.ordering isa PerIndex
end

@testitem "LockstepFunction direct constructor" begin
    using LockstepODE

    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    wrapper = SimpleWrapper(harmonic_oscillator!)
    lockstep_func = LockstepFunction(wrapper, 2, 2, true, PerODE(), nothing)
    @test lockstep_func.num_odes == 2
    @test lockstep_func.ode_size == 2
end
