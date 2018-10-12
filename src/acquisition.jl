abstract type AbstractAcquisition end

mutable struct ProbabilityOfImprovement <: AbstractAcquisition
    τ::Float64
end
function acquisitionfunction(a::ProbabilityOfImprovement, model)
    (x, g) -> begin
        μ, σ = mean_sigma(model, x)
        σ == 0 && return float(μ > a.τ)
        erf((μ - a.τ)/σ)
    end
end
ProbabilityOfImprovement(; τ = -Inf) = ProbabilityOfImprovement(τ)

mutable struct ExpectedImprovement <: AbstractAcquisition
    τ::Float64
end
ExpectedImprovement(; τ = -Inf) = ExpectedImprovement(τ)
function setparams!(a::Union{ExpectedImprovement,ProbabilityOfImprovement}, model)
    a.τ = maxy(model)
end
function acquisitionfunction(a::ExpectedImprovement, model)
    (x, g) -> begin
        μ, σ = mean_sigma(model, x)
        σ == 0 && return μ > a.τ ? μ - a.τ : 0.
        (μ - a.τ) * erf((μ - a.τ)/σ) + σ * 1/√(2π) * exp(-1/2 * ((μ - a.τ)/σ)^2)
    end
end

mutable struct UpperConfidenceBound <: AbstractAcquisition
    ϵ::Float64
    ηt::Float64
end
UpperConfidenceBound(; ϵ = .1, ηt = 1) = UpperConfidenceBound(ϵ, ηt)
function setparams!(a::UpperConfidenceBound, model)
    D, nobs = dims(model)
    a.ηt = sqrt(2*log(nobs^(D/2 + 2)*π^2/(3*a.ϵ)))
end
function acquisitionfunction(a::UpperConfidenceBound, model)
    (x, g) -> begin
        μ, σ = mean_sigma(model, x)
        μ + a.ηt * σ
    end
end

# TODO
mutable struct ThompsonSampling <: AbstractAcquisition
end

# TODO
mutable struct EntropySearch <: AbstractAcquisition
end

# TODO
mutable struct PredictiveEntropySearch <: AbstractAcquisition
end

setparams!(a, model) = nothing

struct MaxMean <: AbstractAcquisition end
acquisitionfunction(a::MaxMean, model) = (x, g) -> mean_sigma(model, x)[1]

### optimize acquisition function
acquire_max(o::BOpt) = acquire_max(o.acquisition, o.model, o.lowerbounds, o.upperbounds)
acquire_model_max(o::BOpt) = acquire_max(MaxMean(), o.model, o.lowerbounds, o.upperbounds)
function acquire_max(a::AbstractAcquisition, model, lowerbounds, upperbounds;
                     method = :GN_AGS, restarts = 1)
    D = length(lowerbounds)
    opt = NLopt.Opt(method, D)
    NLopt.maxeval!(opt, 200)
    NLopt.lower_bounds!(opt, lowerbounds)
    NLopt.upper_bounds!(opt, upperbounds)
    NLopt.max_objective!(opt, acquisitionfunction(a, model))
    maxf = -Inf
    maxx = lowerbounds
    for _ in 1:restarts
        x0 = sample(lowerbounds, upperbounds)
        f, x, ret = NLopt.optimize(opt, x0)
#         println("$x0, $x, $f, $ret")
        if f > maxf
            maxf = f
            maxx = x
        end
    end
    maxf, maxx
end
