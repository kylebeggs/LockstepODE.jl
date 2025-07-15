# Parameter handling functions
function _extract_parameter_prototype(p::AbstractVector{<:AbstractVector}, num_odes::Int)
    @assert length(p) == num_odes "Length of p ($(length(p))) vector must equal num_odes ($num_odes)"
    return p[1]
end
_extract_parameter_prototype(p, ::Int) = p  # Single parameter set

function _get_ode_parameters(p::AbstractVector, i::Int, num_odes::Int)
    length(p) == num_odes ? p[i] : p
end
_get_ode_parameters(p, ::Int, ::Int) = p

recursive_null_parameters(::LockstepFunction{O,F,B}, u, p) where {O,F,B} = nothing

function batch_parameters(lockstep_func::LockstepFunction{O,F,B}, p::AbstractVector{<:AbstractVector}) where {O,F,B}
    @assert length(p) == lockstep_func.num_odes "Length of p vector must equal num_odes"
    return p
end

function batch_parameters(::LockstepFunction{O,F,B}, p) where {O,F,B}
    return p
end

# Index management functions for LockstepFunction
function _get_idxs(lockstep_func::LockstepFunction{PerODE,F,B}, i) where {F,B}
    ((i - 1) * lockstep_func.ode_size + 1):(i * lockstep_func.ode_size)
end

function _get_idxs(lockstep_func::LockstepFunction{PerIndex,F,B}, i) where {F,B}
    i:(lockstep_func.num_odes):((lockstep_func.ode_size * (lockstep_func.num_odes - 1)) + i)
end

# Batching and extraction utilities
function batch_initial_conditions(
    lockstep_func::LockstepFunction{O,F,B}, u0::AbstractVector{T}
) where {O,F,B,T<:Number}
    result = repeat(u0, lockstep_func.num_odes)
    return result
end

function batch_initial_conditions(
    lockstep_func::LockstepFunction{O,F,B}, u0::AbstractVector{<:AbstractVector}
) where {O,F,B}
    @assert length(u0) == lockstep_func.num_odes "Length of u0 vector must equal num_odes"
    result = vcat(u0...)
    return result
end

function extract_solutions(lockstep_func::LockstepFunction{O,F,B}, sol) where {O,F,B}
    individual_sols = map(1:(lockstep_func.num_odes)) do i
        start_idx = (i - 1) * lockstep_func.ode_size + 1
        end_idx = i * lockstep_func.ode_size

        u_series = [Array(u[start_idx:end_idx]) for u in sol.u]

        return (u=u_series, t=sol.t)
    end
    return individual_sols
end
