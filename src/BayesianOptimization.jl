module BayesianOptimization
import NLopt, GaussianProcesses
import GaussianProcesses: GPBase, GPE
import ElasticPDMats: ElasticPDMat
import SpecialFunctions: erf
using ForwardDiff, DiffResults, Random, Dates
export BOpt, ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, 
ThompsonSamplingSimple, boptimize!, MLGPOptimizer, NoOptimizer, Min, Max,
BrochuBetaScaling, NoBetaScaling, Silent, Timings, Progress


mutable struct IterationCounter
    c::Int
    i::Int
    N::Int
end
isdone(s::IterationCounter) = s.c == s.N
step!(s::IterationCounter) = (s.c += 1; s.i += 1)
init!(s::IterationCounter) = s.c = 0
mutable struct DurationCounter
    starttime::Float64
    duration::Float64
    now::Float64
    endtime::Float64
end
function init!(s::DurationCounter)
    s.starttime = time()
    s.endtime = s.starttime + s.duration
end
isdone(s::DurationCounter) = (s.now = time()) >= s.endtime
abstract type ModelOptimizer end
mutable struct MLGPOptimizer{NT} <: ModelOptimizer
    i::Int
    every::Int
    options::NT
end
"""
    MLGPOptimizer(; every = 10, kwargs...)

Set the GP hyperparameters to the maximum likelihood estimate `every` number of steps.
"""
MLGPOptimizer(; every = 10, kwargs...) = MLGPOptimizer(0, every, kwargs.data)
function optimizemodel!(o::MLGPOptimizer, model::GPBase)
    if o.i % o.every == 0
        optimizemodel!(model; o.options...)
    end
    o.i += 1
end

"""
Don't optimize the model ever.
"""
struct NoModelOptimizer <: ModelOptimizer end
optimizemodel!(o::NoModelOptimizer, model) = Nothing

@enum Sense Min=-1 Max=1
@enum Verbosity Silent=0 Timings=1 Progress=2

mutable struct BOpt{F,M,A,MO}
    func::F
    sense::Sense
    model::M
    acquisition::A
    acquisitionoptions::NamedTuple
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
end
isdone(o::BOpt) = isdone(o.iterations) || isdone(o.duration)
"""
    BOpt(func, model, acquisition, modeloptimizer, lowerbounds, upperbounds; 
              sense = Max, maxiterations = 10^4, maxduration = Inf, 
              acquisitionoptions = NamedTuple(), gradientfree = false,
              verbosity = Progress, lhs_iterations = 5*length(lowerbounds))
"""
function BOpt(func, model, acquisition, modeloptimizer, lowerbounds, upperbounds; 
              sense = Max, maxiterations = 10^4, maxduration = Inf, 
              acquisitionoptions = NamedTuple(), gradientfree = false,
              verbosity = Progress, lhs_iterations = 5*length(lowerbounds))
    if gradientfree
        default_acquisitionoptions = (method = :GN_DIRECT_L, restarts = 1, maxeval = 500)
    else
        default_acquisitionoptions = (method = :LD_LBFGS, restarts = 10, maxeval = 500) 
    end
    acquisitionoptions = merge(default_acquisitionoptions, acquisitionoptions)
    now = time()
    BOpt(func, sense, model, acquisition, acquisitionoptions,
         modeloptimizer, lowerbounds, upperbounds,
         -Inf64*Int(sense), Array{Float64}(undef, length(lowerbounds)), 
         -Inf64*Int(sense), Array{Float64}(undef, length(lowerbounds)), 
         IterationCounter(0, 0, maxiterations), 
         DurationCounter(now, maxduration, now, now + maxduration),
         NLopt.Opt(acquisitionoptions.method, length(lowerbounds)),
         verbosity, lhs_iterations)
end
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

sample(lowerbounds, upperbounds) =
    rand(length(lowerbounds)) .* (upperbounds .- lowerbounds) .+ lowerbounds

function initialise_model!(o)
    dac = @elapsed x = latin_hypercube_sampling(o.lowerbounds, o.upperbounds, 
                                                o.lhs_iterations)
    dfunc = @elapsed y = Int(o.sense) .* o.func.([x[:, i] for i in 1:size(x, 2)])
    o.iterations.i = o.iterations.c = length(y)
    dmu = @elapsed update!(o.model, x, y)
    dom = @elapsed optimizemodel!(o.modeloptimizer, o.model)
    o.opt = nlopt_setup(o.acquisition, o.model, o.lowerbounds, o.upperbounds;
                        o.acquisitionoptions...)
    dac, dfunc, dmu, dom
end
"""
    boptimize!(o::BOpt)
"""
function boptimize!(o::BOpt)
    init!(o.duration)
    init!(o.iterations)
    dfunc = dom = dac = dmu = 0.
    if o.iterations.i == 0 dac, dfunc, dmu, dom = initialise_model!(o) end
    while !isdone(o)
        o.verbosity >= Progress && @info("$(now())\titeration: $(o.iterations.i)\tcurrent optimum: $(o.observed_optimum)")
        setparams!(o.acquisition, o.model)
        dac += @elapsed f, x = acquire_max(o.opt, o.lowerbounds, o.upperbounds, 
                                           o.acquisitionoptions.restarts)
        dfunc += @elapsed y = Int(o.sense) * o.func(x)
        step!(o.iterations)
        if y > Int(o.sense) * o.observed_optimum
            o.observed_optimum = Int(o.sense) * y
            o.observed_optimizer = x
        end
        dmu += @elapsed update!(o.model, x, y)
        dom += @elapsed optimizemodel!(o.modeloptimizer, o.model)
    end
    o.duration.now = time() 
    o.verbosity >= Timings && @info("time spent for:
        function evaluation \t $dfunc s
        model update \t\t $dmu s
        model optimization \t $dom s
        acquisition \t\t $dac s")
    dom += @elapsed o.model_optimum, o.model_optimizer = acquire_model_max(o, restarts = 10, maxeval = 2000)
    (observerd_optimum = o.observed_optimum, observed_optimizer = o.observed_optimizer,
     model_optimum = Int(o.sense) * o.model_optimum, model_optimizer = o.model_optimizer)
end

include("acquisition.jl")
include("models/gp.jl")

end # module
