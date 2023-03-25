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
import NLopt, GaussianProcesses, Sobol
import Sobol: SobolSeq
import GaussianProcesses: GPBase, GPE
import ElasticArrays: ElasticArray
using ForwardDiff, DiffResults, Random, Dates, SpecialFunctions, TimerOutputs
export BOpt, ExpectedImprovement, ProbabilityOfImprovement,
UpperConfidenceBound, ThompsonSamplingSimple, MutualInformation, boptimize!,
MAPGPOptimizer, NoModelOptimizer, Min, Max, BrochuBetaScaling, NoBetaScaling,
Silent, Timings, Progress, ScaledSobolIterator, ScaledLHSIterator, maxduration!,
maxiterations!, optimize

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

mutable struct BOpt{F,M,A,AO,MO,Ti}
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
function BOpt(func, model, acquisition, modeloptimizer, lowerbounds, upperbounds;
              sense = Max, maxiterations = 10^4, maxduration = Inf,
              acquisitionoptions = NamedTuple(),
              repetitions = 1, verbosity = Progress,
              initializer_iterations = 5*length(lowerbounds),
              initializer = ScaledSobolIterator(lowerbounds, upperbounds,
                                                initializer_iterations))
    now = time()
    acquisitionoptions = merge(defaultoptions(typeof(model), typeof(acquisition)),
               acquisitionoptions)
    maxiterations < length(initializer) && @error("maxiterations = $maxiterations < length(initializer) = $(length(initializer))")

    current_optimum   = isempty(model.y) ? -Inf*Int(sense)           : Int(sense) * maximum(model.y)
    current_optimizer = isempty(model.y) ? zero(float.(lowerbounds)) : Array(model.x[:, argmax(model.y)])
    BOpt(func, sense, model, acquisition,
         acquisitionoptions,
         modeloptimizer, float.(lowerbounds), float.(upperbounds),
         current_optimum, current_optimizer,
         current_optimum, copy(current_optimizer),
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
        println(io, "duration: $(o.duration.now - o.duration.starttime)/$(o.duration.duration) s")
    end
end

function initialise_model!(o)
    ys = Float64[]
    xs = Vector{Float64}[]
    for x in o.initializer
        for j in 1:o.repetitions
            push!(ys, _evaluate_function(o, x))
            push!(xs, x)
        end
    end
    o.iterations.i = o.iterations.c = length(ys)/o.repetitions
    @mytimeit o.timeroutput "model update" update!(o.model, hcat(xs...), ys)
    @mytimeit o.timeroutput "model hyperparameter optimization" optimizemodel!(o.modeloptimizer, o.model)
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
        o.verbosity >= Progress && @info("$(now())\titeration: $(o.iterations.i)\tcurrent optimum: $(o.observed_optimum)")
        setparams!(o.acquisition, o.model)
        @mytimeit o.timeroutput "acquisition" f, x = acquire_max(o.opt, o.lowerbounds, o.upperbounds, o.acquisitionoptions.restarts)
        ys = Float64[]
        step!(o.iterations)
        for _ in 1:o.repetitions
            y = _evaluate_function(o, x)
            push!(ys, y)
        end
        @mytimeit o.timeroutput "model update" update!(o.model, hcat(fill(x, o.repetitions)...), ys)
        @mytimeit o.timeroutput "model hyperparameter optimization" optimizemodel!(o.modeloptimizer, o.model)
    end
     @mytimeit o.timeroutput "acquisition" o.model_optimum, o.model_optimizer = acquire_model_max(o)
    o.duration.now = time()
    o.verbosity >= Timings && @info(o.timeroutput)
    (observed_optimum = o.observed_optimum, observed_optimizer = o.observed_optimizer,
     model_optimum = Int(o.sense) * o.model_optimum, model_optimizer = o.model_optimizer)
end

function _evaluate_function(o, x)
    @mytimeit o.timeroutput "function evaluation" y = Int(o.sense) * o.func(x)
    if y > Int(o.sense) * o.observed_optimum
        o.observed_optimum = Int(o.sense) * y
        o.observed_optimizer = x
    end
    return y
end
"""
    optimize(f, inputdimension; optkwargs...)

Optimize function `f` using default parameters. All parameters can be overwritten by passing respective values as keyword arguments. Pass dimension of the domain of `f` as a 2nd argument.
    
TODO: document `optkwargs` defaults here, `optkwargs` may include keywords:

model, acquisition, modeloptimizer, lowerbounds, upperbounds, sense, maxiterations, maxduration, acquisitionoptions, repetitions, verbosity, initializer_iterations, initializer
"""
function optimize(f, inputdimension; optkwargs...)
    # TODO: Can inputdimension be avoided, e.g., by inspecting MethodTable of f?
    # TODO: come up with sensible optdefaults, 
    #       maybe move the def. somewhere else

    # optdefaults can be extended to overwrite kwargs in the BOpt constructor 
    optdefaults = (
        func = f,
        model = GaussianProcesses.ElasticGPE(
            inputdimension,
            mean = GaussianProcesses.MeanConst(0.0),
            kernel = GaussianProcesses.Mat52Ard(zeros(inputdimension), 0.0),
            logNoise = -2.0,
            capacity = 3000,
        ),
        acquisition = ExpectedImprovement(),
        modeloptimizer = MAPGPOptimizer(
            every = 20,
            noisebounds = [-4, 3],
            kernbounds = [[-3 * ones(inputdimension); -3], [4 * ones(inputdimension); 3]],
            maxeval = 100,
        ),
        lowerbounds = -100 .* ones(inputdimension),
        upperbounds = 100 .* ones(inputdimension)   #,
        # sense = Min, 
        # ...
    )
    # merging from left-to-right, order of args is kept
    # optkwargs overwrites and extends optdefaults
    params = merge(optdefaults, optkwargs)
    # split params into args of type Array and kwargs of type NamedTuple
    argnames = [:func, :model, :acquisition, :modeloptimizer, :lowerbounds, :upperbounds]
    args = [v for (k, v) in pairs(params) if k in argnames]
    kwargs = (;
        zip(
            [k for (k, v) in pairs(params) if !(k in argnames)],
            [v for (k, v) in pairs(params) if !(k in argnames)],
        )...
    )
    opt = BOpt(args...; kwargs...)
    boptimize!(opt)
end

end # module
