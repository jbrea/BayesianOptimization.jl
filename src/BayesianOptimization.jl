module BayesianOptimization
import MathProgBase, NLopt, GaussianProcesses
import GaussianProcesses: GPBase, GPE
import ElasticPDMats: ElasticPDMat
export BOpt, ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, 
optimize!, GPOptimizer, NoOptimizer, Min, Max


mutable struct IterationCounter
    i::Int64
    N::Int64
end
isdone(s::IterationCounter) = (s.i += 1) == s.N
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
    modeloptimizer::MO
    lowerbounds::Array{Float64, 1}
    upperbounds::Array{Float64, 1}
    fmax::Float64
    xmax::Array{Float64, 1}
    iterations::IterationCounter
    duration::DurationCounter
end
isdone(o::BOpt) = isdone(o.iterations) || isdone(o.duration)
function BOpt(func, model, acquisition, modeloptimizer, lowerbounds, upperbounds; 
              sense = Max, maxiterations = 10^4, maxduration = 200)
    now = time()
    BOpt(func, sense, model, acquisition, modeloptimizer, lowerbounds, upperbounds,
         -Inf64*Int(sense), Array{Float64}(undef, length(lowerbounds)), 
         IterationCounter(0, maxiterations), 
         DurationCounter(now, maxduration, now, now + maxduration))
end
import Base: show
function show(io::IO, mime::MIME"text/plain", o::BOpt)
    println(io, "Bayesian Optimization object\n\nmodel:")
    show(io, mime, o.model)
    println("\nacquisition:")
    show(io, mime, o.acquisition)
    println(io, "\n\noptimum: $(o.fmax)")
    println(io, "optimizer: $(o.xmax)")
    println(io, "iterations: $(o.iterations.i)/$(o.iterations.N)")
    println(io, "duration: $(o.duration.now - o.duration.starttime)/$(o.duration.duration) s")
end

sample(lowerbounds, upperbounds) =
    rand(length(lowerbounds)) .* (upperbounds .- lowerbounds) .+ lowerbounds

function optimize!(o::BOpt)
    x = sample(o.lowerbounds, o.upperbounds)
    init!(o.duration)
    dfunc = 0.
    dom = 0.
    dac = 0.
    dmu = 0.
    while !isdone(o)
        dfunc += @elapsed y = Int(o.sense) * o.func(x)
        if y > Int(o.sense) * o.fmax
            o.fmax = Int(o.sense) * y
            o.xmax = x
        end
        dmu += @elapsed update!(o.model, x, y)
        dom += @elapsed optimizemodel!(o.modeloptimizer, o.model)
        setparams!(o.acquisition, o.model)
        dac += @elapsed f, x = acquire_max(o)
    end
    println("time spent for:")
    println("function evaluation \t $dfunc s")
    println("model update \t\t $dmu s")
    println("model optimization \t $dom s")
    println("acquisition \t\t $dac s")
    model_fmax, model_xmax = acquire_model_max(o)
    (observerd_optimum = o.fmax, observed_optimizer = o.xmax,
     model_optimum = Int(o.sense) * model_fmax, model_optimizer = model_xmax)
end

include("acquisition.jl")
include("model.jl")

end # module
