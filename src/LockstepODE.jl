module LockstepODE

include("types.jl")
include("solve.jl")
include("solution.jl")

# Core types
export LockstepFunction, LockstepSolution

# Solve interface
export solve

# Solution utilities
export extract_solutions, extract_at_time, get_sync_states

end
