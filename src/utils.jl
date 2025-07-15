# Parameter handling functions
function _get_ode_parameters(p::AbstractVector, i::Int, num_odes::Int)
    length(p) == num_odes ? p[i] : p
end
_get_ode_parameters(p, ::Int, ::Int) = p

# Index management functions for LockstepFunction
function _get_idxs(lockstep_func::LockstepFunction{PerODE, F, B}, i) where {F, B}
    ((i - 1) * lockstep_func.ode_size + 1):(i * lockstep_func.ode_size)
end

function _get_idxs(lockstep_func::LockstepFunction{PerIndex, F, B}, i) where {F, B}
    i:(lockstep_func.num_odes):((lockstep_func.ode_size * (lockstep_func.num_odes - 1)) + i)
end

# Batching and extraction utilities
function batch_initial_conditions(
        u0::AbstractVector{T}, num_odes::Int, ode_size::Int
) where {T <: Number}
    expected_batched_size = num_odes * ode_size

    # If u0 is already the correct batched size, return as-is
    if length(u0) == expected_batched_size
        return u0
    elseif length(u0) == ode_size
        # If u0 is a single ODE, repeat it for all ODEs
        result = repeat(u0, num_odes)
        return result
    else
        error("u0 length ($(length(u0))) doesn't match expected single ODE size ($ode_size) or batched size ($expected_batched_size)")
    end
end

function batch_initial_conditions(
        u0::AbstractVector{<:AbstractVector}, num_odes::Int, ::Int
)
    @assert length(u0)==num_odes "Length of u0 vector must equal num_odes"
    result = vcat(u0...)
    return result
end

function extract_solutions(lockstep_func::LockstepFunction{O, F, B}, sol) where {O, F, B}
    individual_sols = map(1:(lockstep_func.num_odes)) do i
        start_idx = (i - 1) * lockstep_func.ode_size + 1
        end_idx = i * lockstep_func.ode_size

        u_series = [Array(u[start_idx:end_idx]) for u in sol.u]

        return (u = u_series, t = sol.t)
    end
    return individual_sols
end
