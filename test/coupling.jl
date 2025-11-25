@testitem "Synchronized solve with coupling" begin
    using LockstepODE
    using OrdinaryDiffEq
    using SciMLBase
    using Statistics

    # Simple decay
    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    # Coupling: average first component
    function couple!(states, t)
        mean_val = mean(s[1] for s in states)
        for s in states
            s[1] = mean_val
        end
    end

    lf = LockstepFunction(decay!, 1, 3;
        sync_interval=0.1,
        coupling=couple!
    )

    u0s = [[1.0], [2.0], [3.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

    @test sol isa LockstepSolution
    @test length(sol.sync_times) > 1
    @test sol.sync_times[1] ≈ 0.0
    @test sol.sync_times[end] ≈ 1.0

    # After coupling, values should converge
    final_states = [s.u[end][1] for s in sol.solutions]
    @test maximum(final_states) - minimum(final_states) < 0.1
end

@testitem "Sync times accuracy" begin
    using LockstepODE
    using OrdinaryDiffEq

    function f!(du, u, p, t)
        du[1] = 1.0  # constant growth
    end

    function couple!(states, t)
        # Just touch the states
        for s in states
            s[1] = s[1]
        end
    end

    lf = LockstepFunction(f!, 1, 2;
        sync_interval=0.25,
        coupling=couple!
    )

    u0s = [[0.0], [0.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

    # Should have sync at 0.0, 0.25, 0.5, 0.75, 1.0
    @test length(sol.sync_times) == 5
    @test sol.sync_times ≈ [0.0, 0.25, 0.5, 0.75, 1.0] atol=1e-10
end

@testitem "get_sync_states" begin
    using LockstepODE
    using OrdinaryDiffEq

    function f!(du, u, p, t)
        du[1] = 1.0
    end

    function couple!(states, t)
        for s in states
            s[1] += 10.0  # Add 10 at each sync
        end
    end

    lf = LockstepFunction(f!, 1, 2;
        sync_interval=0.5,
        coupling=couple!
    )

    u0s = [[0.0], [0.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

    # Get states at first sync (t=0.0)
    states_t0 = get_sync_states(sol, 1)
    @test length(states_t0) == 2

    # Get states at last sync
    states_end = get_sync_states(sol, length(sol.sync_times))
    @test length(states_end) == 2
end

@testitem "Coupling indices" begin
    using LockstepODE
    using OrdinaryDiffEq
    using Statistics

    # 2D system
    function f!(du, u, p, t)
        du[1] = -u[1]
        du[2] = u[1]  # second component grows based on first
    end

    # Only couple first component
    function couple!(states, t)
        mean_val = mean(s[1] for s in states)
        for s in states
            s[1] = mean_val
        end
    end

    lf = LockstepFunction(f!, 2, 3;
        sync_interval=0.1,
        coupling=couple!,
        coupling_indices=[1]  # Only affects first index
    )

    u0s = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

    # First components should converge
    final_x1 = [s.u[end][1] for s in sol.solutions]
    @test maximum(final_x1) - minimum(final_x1) < 0.1

    # Second components diverge based on initial conditions
    final_x2 = [s.u[end][2] for s in sol.solutions]
    @test maximum(final_x2) - minimum(final_x2) > 0.1
end

@testitem "No coupling with sync_interval" begin
    using LockstepODE
    using OrdinaryDiffEq

    function decay!(du, u, p, t)
        du[1] = -u[1]
    end

    # sync_interval but no coupling - should warn and behave like independent
    lf = @test_logs (:warn, r"sync_interval.*no coupling") LockstepFunction(decay!, 1, 3; sync_interval=0.1)

    u0s = [[1.0], [2.0], [3.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

    # Should solve independently (no sync times since no actual coupling)
    @test length(sol.sync_times) == 0
end

@testitem "Large sync interval (spans tspan)" begin
    using LockstepODE
    using OrdinaryDiffEq

    function f!(du, u, p, t)
        du[1] = 1.0
    end

    function couple!(states, t)
        for s in states
            s[1] = 0.0  # Reset to zero
        end
    end

    # sync_interval larger than tspan - should only sync at end
    lf = LockstepFunction(f!, 1, 2;
        sync_interval=10.0,
        coupling=couple!
    )

    u0s = [[0.0], [0.0]]
    sol = LockstepODE.solve(lf, u0s, (0.0, 1.0), Tsit5())

    # Should have sync at 0.0 and 1.0 only
    @test length(sol.sync_times) == 2
    @test sol.sync_times ≈ [0.0, 1.0]
end

@testitem "Coupling with per-ODE parameters" begin
    using LockstepODE
    using OrdinaryDiffEq
    using Statistics

    function decay!(du, u, p, t)
        du[1] = -p * u[1]  # Different decay rates
    end

    function couple!(states, t)
        mean_val = mean(s[1] for s in states)
        for s in states
            s[1] = mean_val
        end
    end

    lf = LockstepFunction(decay!, 1, 3;
        sync_interval=0.2,
        coupling=couple!
    )

    u0s = [[1.0], [1.0], [1.0]]
    ps = [0.5, 1.0, 2.0]  # Different decay rates

    sol = LockstepODE.solve(lf, u0s, (0.0, 2.0), ps, Tsit5())

    # All ODEs should stay synchronized despite different decay rates
    final_vals = [s.u[end][1] for s in sol.solutions]
    @test maximum(final_vals) - minimum(final_vals) < 0.1
end
