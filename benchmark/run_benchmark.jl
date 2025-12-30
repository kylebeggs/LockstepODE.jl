#!/usr/bin/env julia
#=
Benchmark comparing main branch (batched single integrator) vs
v2.0-multi-integrator (multi-integrator architecture).

Usage:
  julia --project=benchmark -t 8 benchmark/run_benchmark.jl
=#

using Printf
using Chairmarks
using OrdinaryDiffEq

# ============================================================================
# Load Main Branch (worktree_main)
# ============================================================================

const BENCH_DIR = @__DIR__
const MAIN_PATH = joinpath(BENCH_DIR, "worktree_main")
const V2_PATH = dirname(BENCH_DIR)

if !isdir(MAIN_PATH)
    error("""
    Worktree not found at $MAIN_PATH
    Create it with: git worktree add benchmark/worktree_main main
    """)
end

# Define paths before module definitions
const MAIN_SRC = joinpath(MAIN_PATH, "src")
const V2_SRC = joinpath(V2_PATH, "src")

# Load main branch components directly
module LockstepMain
    using OrdinaryDiffEq
    using OhMyThreads
    using OrdinaryDiffEq: DiscreteCallback, ContinuousCallback, CallbackSet
    import OrdinaryDiffEq: ODEProblem

    # Access paths from parent module
    const _main_src = Main.MAIN_SRC

    include(joinpath(_main_src, "core.jl"))
    include(joinpath(_main_src, "utils.jl"))
end

# Load v2.0 branch components
module LockstepV2
    using SciMLBase: ReturnCode, ODEFunction
    using OrdinaryDiffEq: ODEProblem, ODESolution
    using OrdinaryDiffEq: init as ode_init, solve as ode_solve
    using OrdinaryDiffEq: solve! as ode_solve!
    using CommonSolve
    import CommonSolve: solve!, init, solve

    # Access paths from parent module
    const _v2_src = Main.V2_SRC

    include(joinpath(_v2_src, "types.jl"))
    include(joinpath(_v2_src, "problem.jl"))
    include(joinpath(_v2_src, "integrator.jl"))
    include(joinpath(_v2_src, "commonsolve.jl"))
    include(joinpath(_v2_src, "solution.jl"))
end

# ============================================================================
# ODE Test Functions
# ============================================================================

"""
Harmonic oscillator: 2D system (used for M=2)
"""
function harmonic_oscillator!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end

"""
Lorenz system: 3D chaotic system (used for M=3)
"""
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

"""
N-dimensional linear decay: scalable system (used for M > 3)
"""
function linear_decay!(du, u, p, t)
    α = p
    @inbounds for i in eachindex(u)
        du[i] = -α * u[i]
    end
end

# ============================================================================
# Benchmark Parameters
# ============================================================================

# Configurable via command line: QUICK=1 julia ...
const QUICK_MODE = get(ENV, "QUICK", "0") == "1"

const ODE_SIZES = QUICK_MODE ? [2, 4] : [2, 4, 8, 16, 32, 64]
const NUM_SYSTEMS = QUICK_MODE ? [10, 100] : [10, 100, 1000, 10_000]
const TSPAN = (0.0, 10.0)
const ALG = Tsit5()

# ============================================================================
# Benchmark Functions
# ============================================================================

"""
Benchmark main branch (single batched integrator)
"""
function benchmark_main(f!, ode_size::Int, num_odes::Int, u0_single, p; warmup=true)
    # Create LockstepFunction (main branch API)
    lf = LockstepMain.LockstepFunction(f!, ode_size, num_odes; internal_threading=true)

    # Batch initial conditions
    u0_batched = LockstepMain.batch_initial_conditions(u0_single, num_odes, ode_size)

    # Batch parameters: main branch expects vector of length num_odes or scalar
    p_batched = p isa Nothing ? nothing : (p isa Number ? p : fill(p, num_odes))

    # Create problem
    prob = LockstepMain.ODEProblem(lf, u0_batched, TSPAN, p_batched; save_everystep=false)

    # Warmup
    if warmup
        solve(prob, ALG)
    end

    # Benchmark
    result = @b solve($prob, $ALG)
    return result
end

"""
Benchmark v2.0 branch (multi-integrator / Ensemble mode)
"""
function benchmark_v2(f!, ode_size::Int, num_odes::Int, u0_single, p; warmup=true)
    # Create LockstepFunction (v2.0 API)
    lf = LockstepV2.LockstepFunction(f!, ode_size, num_odes)

    # Create u0s as vector of vectors
    u0s = [copy(u0_single) for _ in 1:num_odes]

    # Parameters: v2.0 expects vector of per-ODE parameters
    ps = p isa Nothing ? fill(nothing, num_odes) : (p isa Number ? fill(p, num_odes) : fill(p, num_odes))

    # Create problem - explicitly use Ensemble mode for multi-integrator comparison
    prob = LockstepV2.LockstepProblem{LockstepV2.Ensemble}(lf, u0s, TSPAN, ps)

    # Warmup
    if warmup
        LockstepV2.solve(prob, ALG)
    end

    # Benchmark
    result = @b LockstepV2.solve($prob, $ALG)
    return result
end

# ============================================================================
# GPU Benchmark Functions (main branch only)
# ============================================================================

# NOTE: GPU benchmarks require GPU-compatible ODE functions using KernelAbstractions.jl
# The current scalar ODE functions (harmonic_oscillator!, lorenz!, linear_decay!) use
# scalar indexing which doesn't work on GPU arrays. To enable GPU benchmarks:
# 1. Write GPU-compatible ODE kernels using KernelAbstractions.jl
# 2. Load the LockstepODE CUDA extension from main branch
#
# For now, GPU benchmarks are disabled. Focus is on comparing CPU architectures.
const CUDA_AVAILABLE = false

# ============================================================================
# Output Formatting
# ============================================================================

function format_time(t::Float64)
    if t < 1e-6
        return @sprintf("%8.2f ns", t * 1e9)
    elseif t < 1e-3
        return @sprintf("%8.2f μs", t * 1e6)
    elseif t < 1.0
        return @sprintf("%8.2f ms", t * 1e3)
    else
        return @sprintf("%8.2f s ", t)
    end
end

function format_memory(bytes)
    bytes = round(Int, bytes)
    if bytes < 1024
        return @sprintf("%8d B ", bytes)
    elseif bytes < 1024^2
        return @sprintf("%8.2f KB", bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf("%8.2f MB", bytes / 1024^2)
    else
        return @sprintf("%8.2f GB", bytes / 1024^3)
    end
end

function print_header()
    println("=" ^ 100)
    println("LockstepODE Benchmark: main (single-integrator) vs v2.0 (multi-integrator)")
    println("=" ^ 100)
    println()
    println("Configuration:")
    println("  ODE sizes (M): ", ODE_SIZES)
    println("  Num systems (N): ", NUM_SYSTEMS)
    println("  Timespan: ", TSPAN)
    println("  Algorithm: ", ALG)
    println("  Threads: ", Threads.nthreads())
    println()
end

function print_cpu_table(results::Vector)
    println("CPU Benchmarks")
    println("-" ^ 100)
    @printf("%-6s %-8s %15s %15s %12s %12s %10s\n",
            "M", "N", "main (time)", "v2.0 (time)", "main (mem)", "v2.0 (mem)", "speedup")
    println("-" ^ 100)

    for r in results
        speedup = r.main_time / r.v2_time
        speedup_str = speedup > 1 ? @sprintf("%8.2fx", speedup) : @sprintf("%8.2fx", speedup)
        @printf("%-6d %-8d %15s %15s %12s %12s %10s\n",
                r.M, r.N,
                format_time(r.main_time),
                format_time(r.v2_time),
                format_memory(r.main_mem),
                format_memory(r.v2_mem),
                speedup_str)
    end
    println()
end

# ============================================================================
# Main Benchmark Runner
# ============================================================================

function get_ode_config(M::Int)
    if M == 2
        return harmonic_oscillator!, [1.0, 0.0], nothing
    elseif M == 3
        return lorenz!, [1.0, 0.0, 0.0], (10.0, 28.0, 8/3)
    else
        return linear_decay!, ones(M), 0.1
    end
end

function run_benchmarks()
    print_header()

    cpu_results = []

    # CPU benchmarks
    println("Running CPU benchmarks...")
    for M in ODE_SIZES
        for N in NUM_SYSTEMS
            f!, u0, p = get_ode_config(M)

            print("  M=$M, N=$N ... ")

            # Benchmark main branch
            main_result = benchmark_main(f!, M, N, u0, p)

            # Benchmark v2.0 branch
            v2_result = benchmark_v2(f!, M, N, u0, p)

            push!(cpu_results, (
                M = M,
                N = N,
                main_time = main_result.time,
                main_mem = main_result.allocs,
                v2_time = v2_result.time,
                v2_mem = v2_result.allocs
            ))

            println("done")
        end
    end

    println()
    print_cpu_table(cpu_results)

    # GPU benchmarks note
    println("Note: GPU benchmarks disabled (requires KernelAbstractions-compatible ODE functions)")
    println()

    println("=" ^ 100)
    println("Benchmark complete")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
