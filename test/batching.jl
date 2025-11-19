@testitem "batch_initial_conditions with single u0" begin
    using LockstepODE

    u0_single = [1.0, 0.0]
    u0_batched = batch_initial_conditions(u0_single, 3, 2)
    @test u0_batched == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
end

@testitem "batch_initial_conditions with vector of u0s" begin
    using LockstepODE

    u0_vec = [[1.0, 0.0], [2.0, 1.0], [0.0, -1.0]]
    u0_batched_vec = batch_initial_conditions(u0_vec, 3, 2)
    @test u0_batched_vec == [1.0, 0.0, 2.0, 1.0, 0.0, -1.0]
end
