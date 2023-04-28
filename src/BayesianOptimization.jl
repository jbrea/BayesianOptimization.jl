module BayesianOptimization
using NLopt
using GaussianProcesses
using Sobol
using Sobol: SobolSeq
using GaussianProcesses: GPBase, GPE
using ElasticArrays: ElasticArray
using ForwardDiff
using DiffResults
using Random
using Dates
using SpecialFunctions
using TimerOutputs
export BOpt, ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, ThompsonSamplingSimple, MutualInformation, boptimize!, MAPGPOptimizer, NoModelOptimizer, Min, Max, BrochuBetaScaling, NoBetaScaling, Silent, Timings, ScaledSobolIterator, ScaledLHSIterator, maxduration!, maxiterations!, optimize

ENABLE_TIMINGS = true

abstract type ModelOptimizer end

"""
Don't optimize the model ever.
"""
struct NoModelOptimizer <: ModelOptimizer end
optimizemodel!(o::NoModelOptimizer, model) = nothing

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
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}
    observed_optimum::Float64
    observed_optimizer::Vector{Float64}
    model_optimum::Float64
    model_optimizer::Vector{Float64}
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
function BOpt{F<:Function, M<:GPBase, A<:AcquisitionFunction, AO<:NamedTuple, MO<:ModelOptimizer, Ti<:Iterator}(func::F, model::M, acquisition::A, modeloptimizer::MO, lowerbounds::Vector{Float64}, upperbounds::Vector{Float64}; sense::Sense=Max, maxiterations::Int=10^4, maxduration::Real=Inf, acquisitionoptions::AO=NamedTuple(), repetitions::Int=1, verbosity::Verbosity=Progress, initializer_iterations::Int=5*length(lowerbounds), initializer::Ti=ScaledSobolIterator(lowerbounds, upperbounds, initializer_iterations)) where {F, M, A, AO, MO, Ti}
    now
