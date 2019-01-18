module BayesianOptimization
import NLopt, GaussianProcesses
import GaussianProcesses: GPBase, GPE
import ElasticPDMats: ElasticPDMat
using ForwardDiff, DiffResults, Random, Dates, SpecialFunctions, TimerOutputs
export BOpt, ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound,
ThompsonSamplingSimple, MutualInformation, boptimize!, MLGPOptimizer, NoOptimizer,
Min, Max, BrochuBetaScaling, NoBetaScaling, Silent, Timings, Progress

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

mutable struct BOpt{F,M,A,AO,MO}
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
    lhs_iterations::Int
    repetitions::Int
    timeroutput::TimerOutput
end
"""
    BOpt(func, model, acquisition, modeloptimizer, lowerbounds, upperbounds;
              sense = Max, maxiterations = 10^4, maxduration = Inf,
              acquisitionoptions = NamedTuple(), repetitions = 1,
              verbosity = Progress, lhs_iterations = 5*length(lowerbounds))
"""
function BOpt(func, model, acquisition, modeloptimizer, lowerbounds, upperbounds;
              sense = Max, maxiterations = 10^4, maxduration = Inf,
              acquisitionoptions = NamedTuple(),
              repetitions = 1, verbosity = Progress,
              lhs_iterations = 5*length(lowerbounds))
    now = time()
    acquisitionoptions = merge(defaultoptions(typeof(model), typeof(acquisition)),
               acquisitionoptions)
    maxiterations < lhs_iterations && @error("maxiterations = $maxiterations < lhs_iterations = $lhs_iterations")
    BOpt(func, sense, model, acquisition,
         acquisitionoptions,
         modeloptimizer, lowerbounds, upperbounds,
         -Inf64*Int(sense), Array{Float64}(undef, length(lowerbounds)),
         -Inf64*Int(sense), Array{Float64}(undef, length(lowerbounds)),
         IterationCounter(0, 0, maxiterations),
         DurationCounter(now, maxduration, now, now + maxduration),
         NLopt.Opt(acquisitionoptions.method, length(lowerbounds)),
         verbosity, lhs_iterations, repetitions, TimerOutput())
end
isdone(o::BOpt) = isdone(o.iterations) || isdone(o.duration)
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
    @mytimeit o.timeroutput "acquisition" x = latin_hypercube_sampling(o.lowerbounds, o.upperbounds, o.lhs_iterations)
    y = Float64[]
    for i in 1:size(x, 2)
        for j in 1:o.repetitions
            @mytimeit o.timeroutput "function evaluation" push!(y, Int(o.sense) * o.func(x[:, i]))
        end
    end
    o.iterations.i = o.iterations.c = length(y)/o.repetitions
    @mytimeit o.timeroutput "model update" update!(o.model,
                           hcat(hcat([fill(x[:, i], o.repetitions) for i in 1:size(x, 2)]...)...),
                           y)
    @mytimeit o.timeroutput "model hyperparameter optimization" optimizemodel!(o.modeloptimizer, o.model)
    o.opt = nlopt_setup(o.acquisition, o.model, o.lowerbounds, o.upperbounds,
                        o.acquisitionoptions)
end
"""
    boptimize!(o::BOpt)
"""
function boptimize!(o::BOpt)
    init!(o.duration)
    init!(o.iterations)
    reset_timer!(o.timeroutput)
    o.iterations.i == 0 && initialise_model!(o)
    while !isdone(o)
        o.verbosity >= Progress && @info("$(now())\titeration: $(o.iterations.i)\tcurrent optimum: $(o.observed_optimum)")
        setparams!(o.acquisition, o.model)
        @mytimeit o.timeroutput "acquisition" f, x = acquire_max(o.opt, o.lowerbounds, o.upperbounds, o.acquisitionoptions.restarts)
        ys = Float64[]
        step!(o.iterations)
        for _ in 1:o.repetitions
            @mytimeit o.timeroutput "function evaluation" y = Int(o.sense) * o.func(x)
            push!(ys, y)
            if y > Int(o.sense) * o.observed_optimum
                o.observed_optimum = Int(o.sense) * y
                o.observed_optimizer = x
            end
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

end # module
