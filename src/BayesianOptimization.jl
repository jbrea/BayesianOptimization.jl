"""
This package exports
* `BOpt`, `boptimize!`, `optimize`
* acquisition types: `ExpectedImprovement`, `ProbabilityOfImprovement`, `UpperConfidenceBound`, `ThompsonSamplingSimple`, `MutualInformation`
* scaling of standard deviation in `UpperConfidenceBound`: `BrochuBetaScaling`, `NoBetaScaling`
* GP hyperparameter optimizer: `MAPGPOptimizer`, `NoModelOptimizer`
* Initializer: `ScaledSobolIterator`, `ScaledLHSIterator`
* optimization sense: `Min`, `Max`
* verbosity levels: `Silent`, `Timings`, `Progress`
* helper: maxduration!, maxiterations!

Use the REPL help, e.g. `?Bopt`, to get more information.
"""
module BayesianOptimization
using Random
using Statistics
using StatsBase
using Optimization
using EasyGPs
using Distances
using OptimizationOptimJL
using OptimizationNLopt
using SurrogatesBase
using CommonSolve
using SciMLBase
using ConcreteStructs
using AbstractGPs
using QuasiMonteCarlo
import NLopt, GaussianProcesses, Sobol
import Sobol: SobolSeq
import GaussianProcesses: GPBase, GPE
import ElasticArrays: ElasticArray
using ForwardDiff, DiffResults, Random, Dates, SpecialFunctions, TimerOutputs
export BOpt,
       ExpectedImprovement,
       ProbabilityOfImprovement,
       UpperConfidenceBound,
       ThompsonSamplingSimple,
       MutualInformation,
       boptimize!,
       MAPGPOptimizer,
       NoModelOptimizer,
       Min,
       Max,
       BrochuBetaScaling,
       NoBetaScaling,
       Silent,
       Timings,
       Progress,
       ScaledSobolIterator,
       ScaledLHSIterator,
       maxduration!,
       maxiterations!,
       optimize

ENABLE_TIMINGS = true

abstract type ModelOptimizer end
"""
Don't optimize the model ever.
"""
struct NoModelOptimizer <: ModelOptimizer end
optimizemodel!(o::NoModelOptimizer, model) = Nothing

include("utils.jl")
include("acquisitionfunctions.jl")
include("acquisition.jl")
include("models/gp.jl")

@enum Sense Min=-1 Max=1
@enum Verbosity Silent=0 Timings=1 Progress=2

mutable struct BOpt{F, M, A, AO, MO, Ti}
    func::F
    sense::Sense
    model::M
    acquisition::A
    acquisitionoptions::AO
    modeloptimizer::MO
    lowerbounds::Array{Float64, 1}
    upperbounds::Array{Float64, 1}
    observed_optimum::Float64
    observed_optimizer::Array{Float64, 1}
    model_optimum::Float64
    model_optimizer::Array{Float64, 1}
    iterations::IterationCounter
    duration::DurationCounter
    opt::NLopt.Opt
    verbosity::Verbosity
    initializer::Ti
    repetitions::Int
    timeroutput::TimerOutput
end
"""
    BOpt(func, model, acquisition, modeloptimizer, lowerbounds, upperbounds;
    sense = Max, maxiterations = 10^4, maxduration = Inf,
    acquisitionoptions = NamedTuple(), repetitions = 1,
    verbosity = Progress,
    initializer_iterations = 5*length(lowerbounds),
    initializer = ScaledSobolIterator(lowerbounds, upperbounds,
    initializer_iterations))
"""
function BOpt(func,
              model,
              acquisition,
              modeloptimizer,
              lowerbounds,
              upperbounds;
              sense = Max,
              maxiterations = 10^4,
              maxduration = Inf,
              acquisitionoptions = NamedTuple(),
              repetitions = 1,
              verbosity = Progress,
              initializer_iterations = 5 * length(lowerbounds),
              initializer = ScaledSobolIterator(lowerbounds, upperbounds,
                                                initializer_iterations))
    now = time()
    acquisitionoptions = merge(defaultoptions(typeof(model), typeof(acquisition)),
                               acquisitionoptions)
    maxiterations < length(initializer) &&
        throw(ArgumentError("maxiterations = $maxiterations < length(initializer) = $(length(initializer))"))

    maxiterations >= 0 || throw(ArgumentError("maxiterations < 0"))
    maxduration >= 0 || throw(ArgumentError("maxduration < 0"))
    length(lowerbounds) == length(upperbounds) ||
        throw(ArgumentError("length of lowerbounds does not match length of upperbounds"))
    all(lowerbounds .<= upperbounds) ||
        throw(ArgumentError("lowerbounds are not pointwise less than or eqal to upperbounds, they were possibly passed in the wrong order"))

    current_optimum = isempty(model.y) ? -Inf * Int(sense) : Int(sense) * maximum(model.y)
    current_optimizer = isempty(model.y) ? zero(float.(lowerbounds)) :
                        Array(model.x[:, argmax(model.y)])
    BOpt(func,
         sense,
         model,
         acquisition,
         acquisitionoptions,
         modeloptimizer,
         float.(lowerbounds),
         float.(upperbounds),
         current_optimum,
         current_optimizer,
         current_optimum,
         copy(current_optimizer),
         IterationCounter(0, 0, maxiterations),
         DurationCounter(now, maxduration, now, now + maxduration),
         nlopt_setup(acquisition, model, lowerbounds, upperbounds, acquisitionoptions),
         verbosity, initializer, repetitions, TimerOutput())
end
isdone(o::BOpt) = isdone(o.iterations) || isdone(o.duration)
maxduration!(o::BOpt, d) = maxduration!(o.duration, d)
maxiterations!(o::BOpt, N) = maxiterations!(o.iterations, N)
import Base: show
function show(io::IO, mime::MIME"text/plain", o::BOpt)
    println(io, "Bayesian Optimization object\n\nmodel:")
    show(io, mime, o.model)
    println("\nacquisition:")
    show(io, mime, o.acquisition)
    if o.iterations.i == 0
        println("\nNo observation data.")
    else
        println(io, "\n\nobserved optimum: $(o.observed_optimum)")
        println(io, "observed optimizer: $(o.observed_optimizer)")
        println(io, "model optimum: $(o.model_optimum)")
        println(io, "model optimizer: $(o.model_optimizer)")
        println(io, "iterations: $(o.iterations.i)/$(o.iterations.N)")
        println(io,
                "duration: $(o.duration.now - o.duration.starttime)/$(o.duration.duration) s")
    end
end

function initialise_model!(o)
    ys = Float64[]
    xs = Vector{Float64}[]
    for x in o.initializer
        for j in 1:(o.repetitions)
            push!(ys, _evaluate_function(o, x))
            push!(xs, x)
        end
    end
    o.iterations.i = o.iterations.c = length(ys) / o.repetitions
    @mytimeit o.timeroutput "model update" update!(o.model, hcat(xs...), ys)
    @mytimeit o.timeroutput "model hyperparameter optimization" optimizemodel!(o.modeloptimizer,
                                                                               o.model)
end
"""
    boptimize!(o::BOpt)
"""
function boptimize!(o::BOpt)
    init!(o.duration)
    init!(o.iterations)
    reset_timer!(o.timeroutput)
    o.iterations.i == 0 && length(o.initializer) > 0 && initialise_model!(o)
    while !isdone(o)
        o.verbosity >= Progress &&
            @info("$(now())\titeration: $(o.iterations.i)\tcurrent optimum: $(o.observed_optimum)")
        setparams!(o.acquisition, o.model)
        @mytimeit o.timeroutput "acquisition" f, x=acquire_max(o.opt, o.lowerbounds,
                                                               o.upperbounds,
                                                               o.acquisitionoptions.restarts)
        ys = Float64[]
        step!(o.iterations)
        for _ in 1:(o.repetitions)
            y = _evaluate_function(o, x)
            push!(ys, y)
        end
        @mytimeit o.timeroutput "model update" update!(o.model,
                                                       hcat(fill(x, o.repetitions)...),
                                                       ys)
        @mytimeit o.timeroutput "model hyperparameter optimization" optimizemodel!(o.modeloptimizer,
                                                                                   o.model)
    end
    @mytimeit o.timeroutput "acquisition" o.model_optimum, o.model_optimizer=acquire_model_max(o)
    o.duration.now = time()
    o.verbosity >= Timings && @info(o.timeroutput)
    (observed_optimum = o.observed_optimum,
     observed_optimizer = o.observed_optimizer,
     model_optimum = Int(o.sense) * o.model_optimum,
     model_optimizer = o.model_optimizer)
end

function _evaluate_function(o, x)
    @mytimeit o.timeroutput "function evaluation" y=Int(o.sense) * o.func(x)
    if y > Int(o.sense) * o.observed_optimum
        o.observed_optimum = Int(o.sense) * y
        o.observed_optimizer = x
    end
    return y
end
"""
    optimize(f, lowerbounds, upperbounds; <keyword arguments>)

Find an optimizer between `lowerbounds` and `upperbounds` of a function `f` using default parameters.

Default parameters can be overwritten by passing respective values as keyword arguments.

# Arguments

TODO: document defaults, keywords:
model, acquisition, modeloptimizer, sense, maxiterations, maxduration, acquisitionoptions,
repetitions, verbosity, initializer_iterations, initializer
"""
function optimize(f, lowerbounds, upperbounds; optkwargs...)
    args, kwargs = merge_with_defaults(f, lowerbounds, upperbounds, optkwargs)
    opt = BOpt(args...; kwargs...)
    boptimize!(opt)
end
"""
    merge_with_defaults(f, lowerbounds, upperbounds, optkwargs)
"""
function merge_with_defaults(f, lowerbounds, upperbounds, optkwargs)
    # the same order of args as in BOpt constructor
    args_keys = (:model, :acquisition, :modeloptimizer)
    kwargs_keys = (:sense,
                   :maxiterations,
                   :maxduration,
                   :acquisitionoptions,
                   :repetitions,
                   :verbosity,
                   :initializer_iterations,
                   :initializer)
    # check if optkwargs contains only valid keyword arguments
    issubset(keys(optkwargs), union(args_keys, kwargs_keys)) ||
        throw(ArgumentError("use of unsupported keyword arguments"))

    length(lowerbounds) == length(upperbounds) ||
        throw(ArgumentError("length of lowerbounds does not match length of upperbounds"))
    inputdimension = length(lowerbounds)

    # TODO: come up with reasonable optdefaults
    # optdefaults can be extended to eventually overwrite default kwargs when passed to BOpt constructor
    optdefaults = (model = GaussianProcesses.ElasticGPE(inputdimension,
                                                        mean = GaussianProcesses.MeanConst(0.0),
                                                        kernel = GaussianProcesses.Mat52Ard(zeros(inputdimension),
                                                                                            0.0),
                                                        logNoise = -2.0,
                                                        capacity = 3000),
                   acquisition = ExpectedImprovement(),
                   modeloptimizer = MAPGPOptimizer(every = 20,
                                                   noisebounds = [-4, 3],
                                                   kernbounds = [
                                                       [-3 * ones(inputdimension); -3],
                                                       [4 * ones(inputdimension); 3],
                                                   ],
                                                   maxeval = 100),
                   maxiterations = 10^3
                   # ...
                   )
    # check if configuration of optdefaults uses valid keywords
    @assert issubset(keys(optdefaults), union(args_keys, kwargs_keys))
    "optdefaults contains unsupported keyword arguments"
    @assert issubset(args_keys, keys(optdefaults))
    "optdefaults has to contain default values for at least: model, acquisition, modeloptimizer, i.e., those not having defaults in BOpt constr."

    # merging from left-to-right, order of args is kept
    # optkwargs overwrites and extends optdefaults
    params = merge(optdefaults, optkwargs)
    # split params into args and kwargs for BOpt constructor, args are in the right order for BOpt constructor
    args = (f, [params[k] for k in args_keys]..., lowerbounds, upperbounds)
    kwargs = NamedTuple((k, v) for (k, v) in pairs(params) if k in kwargs_keys)
    args, kwargs
end



###
### NEW API
###

Base.@kwdef @concrete struct BasicBO
    surrogate
    acquisition_function = OptimizationFunction(UpperConfidenceBound(),
                                                AutoForwardDiff())
    acquisition_optimizer = QuasiMonteCarloMultiStarter()
    hyperparameter_optimizer = Optim.LBFGS(alphaguess = Optim.LineSearches.InitialStatic(; scaled = true),
                                           linesearch = Optim.LineSearches.BackTracking())
    initializer = QMC()
end

function CommonSolve.solve(prob::OptimizationProblem, alg::BasicBO; maxiters = 100)
    acquisition_problem = _acquisition_problem(prob, alg)
    @info "Initializing."
    sign = prob.sense == MaxSense ? 1 : -1
    initialize!(alg.surrogate, prob, alg.initializer)
    acq_alg, acq_kwargs = unpack(alg.acquisition_optimizer)
    for iter in 1:maxiters - length(alg.surrogate.y)
        setparams!(alg.acquisition_function, alg.surrogate)
        @info "Updating hyperparameters."
        update_hyperparameters!(alg.surrogate, alg.hyperparameter_optimizer)
        @info "Acquire new input."
        s = solve(acquisition_problem, acq_alg; acq_kwargs...)
        new_x = s.u
        new_y = prob.f(new_x, prob.p)
        @show (; new_x, new_y)
        SurrogatesBase.update!(alg.surrogate, [new_x], [sign*new_y])
#         callback(state, loss_val, other_args)
    end
end

# SurrogateAbstractGPs

Base.@kwdef @concrete mutable struct SurrogateAbstractGPs <: AbstractStochasticSurrogate
    gp
    noise_variance = .1
    posterior = posterior(gp([[0.]], .1), [0.]) # dummy data
    x = Vector{Float64}[]
    y = Float64[]
end
function SurrogatesBase.finite_posterior(s::SurrogateAbstractGPs, xs)
    s.posterior(xs, s.noise_variance)
end
function SurrogatesBase.update!(s::SurrogateAbstractGPs, new_xs, new_ys)
    append!(s.x, new_xs)
    append!(s.y, new_ys)
    s.posterior = posterior(s.gp(s.x, s.noise_variance), s.y)
    s
end
# TODO: Better. This may not work well without priors or bounds
function SurrogatesBase.update_hyperparameters!(s::SurrogateAbstractGPs, optimizer)
    newgp = EasyGPs.fit(with_gaussian_noise(s.gp, s.noise_variance),
                        s.x, s.y; optimizer)
    s.gp = newgp.gp
    s.noise_variance = newgp.obs_noise
    s
end

maxy(s::SurrogateAbstractGPs) = maximum(s.y)
midpoints(s::SurrogateAbstractGPs) = midpoints(s.x)
dims(s::SurrogateAbstractGPs) = (length(s.x[1]), length(s.x))

# QuasiMonteCarlo

@concrete struct QMC
    n_samples
    factor
    base
    sampler
end
function _sample(sampler::QMC, lb, ub)
    d = length(lb)
    n = if sampler.n_samples === nothing
            sampler.base ^ ceil(Int, log(sampler.base, sampler.factor * d))
        else
            sampler.n_samples
        end
    eachslice(QuasiMonteCarlo.sample(n, lb, ub,
                                     sampler.sampler), dims = 2)
end
function QMC(;
        n_samples = nothing,
        factor = 5,
        base = 2,
        randomizer = MatousekScramble(; base),
        sampler = SobolSample(R = randomizer))
    if n_samples !== nothing
        m = round(Int, log(base, n_samples))
        base^m == n_samples || throw(ArgumentError("n_samples = $n_samples is not a power of base = $base."))
    end
    QMC(n_samples, factor, base, sampler)
end

# Initialization

initialize!(surrogate, initializer::Nothing) = nothing
function initialize!(surrogate, prob, initializer::QMC)
    new_x = _sample(initializer, prob.lb, prob.ub)
    new_y = prob.f.(new_x, Ref(prob.p))
    sign = prob.sense == MaxSense ? 1 : -1
    SurrogatesBase.update!(surrogate, new_x, sign*new_y)
end

# Acquistion

function _acquisition_problem(prob, alg)
    OptimizationProblem(alg.acquisition_function,
                        prob.u0, alg.surrogate,
                        lb = prob.lb, ub = prob.ub,
                        sense = MaxSense)
end

Base.@kwdef @concrete struct QuasiMonteCarloMultiStarter
    optimizer = NLopt.LN_NELDERMEAD()
    sampler = QMC()
end

Base.@kwdef @concrete struct MidPointMultiStarter
    optimizer = NLopt.LN_NELDERMEAD()
end

midpoint(x1, x2) = (x1 + x2) ./ 2
function midpoints(x)
    dists = sort([(Euclidean()(x[i], x[j]), i, j) for i in eachindex(x), j in eachindex(x) if j > i])
    midpoints = [midpoint(x[dists[1][2]], x[dists[1][3]])]
    for d in Iterators.drop(dists, 1)
        m = midpoint(x[d[2]], x[d[3]])
        if minimum(Euclidean()(m, xi) for xi in x) ≥ d[1]/2 &&
            minimum(Euclidean()(m, mi) for mi in midpoints) ≥ d[1]/2
            push!(midpoints, m)
        end
    end
    midpoints
end

startpoints(prob, ::MidPointMultiStarter) = midpoints(prob.p)
function startpoints(prob, alg::QuasiMonteCarloMultiStarter)
    _sample(alg.sampler, prob.lb, prob.ub)
end
function CommonSolve.solve(prob::OptimizationProblem,
        alg::Union{MidPointMultiStarter,QuasiMonteCarloMultiStarter}; kwargs...)
    x_best = copy(prob.u0)
    f_best = prob.sense == MaxSense ? -Inf : Inf
    local_alg, kwargs = unpack(alg.optimizer)
#     cache = Optimization.init(prob, local_alg; kwargs...)
    for m in startpoints(prob, alg)
#         reinit!(cache, u0 = m)
#         u = solve!(cache).u
        prob.u0 .= m
        u = solve(prob, local_alg; kwargs...).u
        # OptimizationSolution.objective has the wrong sign for Optim
        objective = prob.f(u, prob.p)
        if prob.sense == MaxSense && objective > f_best
            f_best = objective
            x_best .= u
        elseif prob.sense == MinSense && objective < f_best
            f_best = objective
            x_best .= u
        end
    end
    (u = x_best, objective = f_best)
end

# HELPER

unpack(t) = t, NamedTuple()
unpack(t::Tuple{<:Any, <:NamedTuple}) = t[1], t[2]

setparams!(f::OptimizationFunction, surrogate) = setparams!(f.f, surrogate)

# FIXES

struct NLoptFailSafe{T}
    opt::T
end
SciMLBase.allowsbounds(o::NLoptFailSafe) = SciMLBase.allowsbounds(o.opt)
SciMLBase.supports_opt_cache_interface(o::NLoptFailSafe) = SciMLBase.supports_opt_cache_interface(o.opt)
function SciMLBase.solve(prob::OptimizationProblem, alg::NLoptFailSafe; kwargs...)
    try
        solve(prob, alg.opt; kwargs...)
    catch e
        @warn "$e; returning u0."
        (; u = prob.u0)
    end
end


end # module
