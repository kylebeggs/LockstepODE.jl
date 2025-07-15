using LockstepODE
import KernelAbstractions as KA
using Test
using OrdinaryDiffEq: ODEProblem, Tsit5, solve

@testset "LockstepODE.jl" begin
    # Simple harmonic oscillator: dx/dt = v, dv/dt = -x
    function harmonic_oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end
    
    @testset "LockstepFunction constructor" begin
        # Test convenient constructor
        lockstep_func = LockstepFunction(harmonic_oscillator!, 2, 2)
        @test lockstep_func.num_odes == 2
        @test lockstep_func.ode_size == 2
        @test lockstep_func.internal_threading == true
        @test lockstep_func.ordering isa PerODE
        @test lockstep_func.f === harmonic_oscillator!
        
        # Test with options
        lockstep_func2 = LockstepFunction(harmonic_oscillator!, 2, 2; internal_threading=false, ordering=PerIndex())
        @test lockstep_func2.internal_threading == false
        @test lockstep_func2.ordering isa PerIndex
        
        # Test direct constructor
        lockstep_func3 = LockstepFunction(harmonic_oscillator!, 2, 2, true, PerODE(), KA.CPU())
        @test lockstep_func3.num_odes == 2
        @test lockstep_func3.ode_size == 2
    end
    
    @testset "batch_initial_conditions" begin
        # Test single initial condition
        u0_single = [1.0, 0.0]
        u0_batched = batch_initial_conditions(u0_single, 3, 2)
        @test u0_batched == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        
        # Test vector of initial conditions
        u0_vec = [[1.0, 0.0], [2.0, 1.0], [0.0, -1.0]]
        u0_batched_vec = batch_initial_conditions(u0_vec, 3, 2)
        @test u0_batched_vec == [1.0, 0.0, 2.0, 1.0, 0.0, -1.0]
    end
    
    
    @testset "LockstepFunction callable" begin
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
    
    @testset "Standard OrdinaryDiffEq.jl workflow" begin
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
    
    @testset "Backend Detection and GPU Support" begin
        u0 = [1.0, 0.0, 2.0, 0.0]
        
        # Test CPU backend detection
        lockstep_func_cpu = LockstepFunction(harmonic_oscillator!, 2, 2)
        @test lockstep_func_cpu.backend isa KA.CPU
        
        # Test explicit backend setting
        lockstep_func_explicit = LockstepFunction(harmonic_oscillator!, 2, 2; backend=KA.CPU())
        @test lockstep_func_explicit.backend isa KA.CPU
        
        # Test backend detection utility
        @test KA.get_backend(u0) isa KA.CPU
    end
    
    
    
end
