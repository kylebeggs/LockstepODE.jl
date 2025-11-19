@testitem "LockstepFunction callable interface" begin
    using LockstepODE

    # Simple harmonic oscillator: dx/dt = v, dv/dt = -x
    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    lockstep_func = LockstepFunction(harmonic_oscillator!, 2, 2)

    # Test callable with manually batched u0
    u0 = [1.0, 0.0, 2.0, 0.0]
    du = zeros(4)
    lockstep_func(du, u0, nothing, 0.0)
    @test du[1] ≈ 0.0  # du[1] = u[2] = 0.0
    @test du[2] ≈ -1.0  # du[2] = -u[1] = -1.0
    @test du[3] ≈ 0.0  # du[3] = u[4] = 0.0
    @test du[4] ≈ -2.0  # du[4] = -u[3] = -2.0
end
