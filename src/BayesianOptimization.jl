module BayesianOptimization
import MathProgBase, NLopt, GaussianProcesses
import GaussianProcesses: GPBase, GPE
import ElasticPDMats: ElasticPDMat
import SpecialFunctions: erf
using ForwardDiff, DiffResults, Random
export BOpt, ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, 
optimize!, GPOptimizer, NoOptimizer, Min, Max


mutable struct IterationCounter
    c::Int64
    i::Int64
    N::Int64
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
mutable struct GPOptimizer{NT} <: ModelOptimizer
    i::Int64
    every::Int64
    options::NT
end
GPOptimizer(; every = 10, kwargs...) = GPOptimizer(0, every, kwargs.data)
function optimizemodel!(o::GPOptimizer, model::GPBase)
    o.i += 1
    if o.i % o.every == 0
        GaussianProcesses.optimize!(model; o.options...)
    end
end

struct NoOptimizer <: ModelOptimizer end
optimizemodel!(o::NoOptimizer, model) = Nothing

@enum Sense Min=-1 Max=1

mutable struct BOpt{F,M,A,MO}
    func::F
    sense::Sense
    model::M
    acquisition::A
    acquisitionoptions::NamedTuple
    modeloptimizer::MO
    lowerbounds::Array{Float64, 1}
    upperbounds::Array{Float64, 1}
    optimum::Float64
    optimizer::Array{Float64, 1}
    model_optimum::Float64
    model_optimizer::Array{Float64, 1}
    iterations::IterationCounter
    duration::DurationCounter
    opt::NLopt.Opt
end
isdone(o::BOpt) = isdone(o.iterations) || isdone(o.duration)
function BOpt(func, model, acquisition, modeloptimizer, lowerbounds, upperbounds; 
              sense = Max, maxiterations = 10^4, maxduration = 200, 
              acquisitionoptions = NamedTuple(), gradientfree = false)
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
         NLopt.Opt(acquisitionoptions.method, length(lowerbounds)))
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
        println(io, "\n\nobserved optimum: $(o.optimum)")
        println(io, "observed optimizer: $(o.optimizer)")
        println(io, "model optimum: $(o.model_optimum)")
        println(io, "model optimizer: $(o.model_optimizer)")
        println(io, "iterations: $(o.iterations.i)/$(o.iterations.N)")
        println(io, "duration: $(o.duration.now - o.duration.starttime)/$(o.duration.duration) s")
    end
end

sample(lowerbounds, upperbounds) =
    rand(length(lowerbounds)) .* (upperbounds .- lowerbounds) .+ lowerbounds

function optimize!(o::BOpt)
    init!(o.duration)
    init!(o.iterations)
    dfunc = 0.
    dom = 0.
    dac = 0.
    dmu = 0.
    x = latin_hypercube_sampling(o.lowerbounds, o.upperbounds, 5*length(o.lowerbounds))
    y = Int(o.sense) .* o.func.([x[:, i] for i in 1:size(x, 2)])
    o.iterations.i = o.iterations.c = length(y)
    update!(o.model, x, y)
    o.opt = nlopt_setup(o.acquisition, o.model, o.lowerbounds, o.upperbounds;
                        o.acquisitionoptions...)
    while !isdone(o)
        setparams!(o.acquisition, o.model)
        dac += @elapsed f, x = acquire_max(o.opt, o.lowerbounds, o.upperbounds, 
                                           o.acquisitionoptions.restarts)
        dfunc += @elapsed y = Int(o.sense) * o.func(x)
        step!(o.iterations)
        if y > Int(o.sense) * o.optimum
            o.optimum = Int(o.sense) * y
            o.optimizer = x
        end
        dmu += @elapsed update!(o.model, x, y)
        dom += @elapsed optimizemodel!(o.modeloptimizer, o.model)
    end
    o.duration.now = time() 
    @info("time spent for:
        function evaluation \t $dfunc s
        model update \t\t $dmu s
        model optimization \t $dom s
        acquisition \t\t $dac s")
    o.model_optimum, o.model_optimizer = acquire_model_max(o, restarts = 10, maxeval = 2000)
    (observerd_optimum = o.optimum, observed_optimizer = o.optimizer,
     model_optimum = Int(o.sense) * o.model_optimum, model_optimizer = o.model_optimizer)
end

include("acquisition.jl")
include("model.jl")

end # module
