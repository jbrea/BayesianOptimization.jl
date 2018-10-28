abstract type AbstractAcquisition end

mutable struct ProbabilityOfImprovement <: AbstractAcquisition
    τ::Float64
end
normal_pdf(μ, σ2) = 1/√(2π*σ2) * exp(-μ^2/(2*σ2))
normal_cdf(μ, σ2) = 1/2 * (1 + erf(μ/√(2σ2)))
function acquisitionfunction(a::ProbabilityOfImprovement, model)
    x -> begin
        μ, σ2 = mean_var(model, x)
        σ2 == 0 && return float(μ > a.τ)
        normal_cdf(μ - a.τ, σ2)
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
    x -> begin
        μ, σ2 = mean_var(model, x)
        σ2 == 0 && return μ > a.τ ? μ - a.τ : 0.
        (μ - a.τ) * normal_cdf(μ - a.τ, σ2) + √σ2 * normal_pdf(μ - a.τ, σ2) 
    end
end

mutable struct UpperConfidenceBound <: AbstractAcquisition
    ϵ::Float64
    ηt::Float64
end
UpperConfidenceBound(; ϵ = .1, ηt = 1) = UpperConfidenceBound(ϵ, ηt)
function setparams!(a::UpperConfidenceBound, model)
    D, nobs = dims(model)
    nobs == 0 && (nobs = 1)
    a.ηt = sqrt(2*log(nobs^(D/2 + 2)*π^2/(3*a.ϵ)))
end
function acquisitionfunction(a::UpperConfidenceBound, model)
    x -> begin
        μ, σ2 = mean_var(model, x)
        μ + a.ηt * √σ2
    end
end

# TODO
mutable struct EntropySearch <: AbstractAcquisition
end

# TODO
mutable struct PredictiveEntropySearch <: AbstractAcquisition
end

setparams!(a, model) = nothing

struct MaxMean <: AbstractAcquisition end
acquisitionfunction(a::MaxMean, model) = x -> mean_var(model, x)[1]

function wrap_gradient(f)
    (x, g) -> begin
        res = DiffResults.DiffResult(0., g)
        ForwardDiff.gradient!(res, f, x)
        res.value
    end
end
wrap_dummygradient(f) = (x, g) -> f(x)

function nlopt_setup(a::AbstractAcquisition, model, lowerbounds, upperbounds;
                     method = :LD_LBFGS, maxeval = 1000, kwargs...)
    D = length(lowerbounds)
    opt = NLopt.Opt(method, D)
    NLopt.maxeval!(opt, maxeval)
    NLopt.lower_bounds!(opt, lowerbounds)
    NLopt.upper_bounds!(opt, upperbounds)
    setparams!(a, model)
    if string(method)[2] == 'D'
        f = wrap_gradient(acquisitionfunction(a, model))
        NLopt.max_objective!(opt, f)
        g = similar(lowerbounds)
        f(sample(lowerbounds, upperbounds), g)
        isnan(g[1]) && @warn("Gradient evaluates to $g. Consider gradient-free methods.")
    else
        NLopt.max_objective!(opt, wrap_dummygradient(acquisitionfunction(a, model)))
    end
    opt
end


### optimize acquisition function
acquire_max(o::BOpt; kwargs...) = acquire_max(o.acquisition, o.model, o.lowerbounds, o.upperbounds; o.acquisitionoptions..., kwargs...)
acquire_model_max(o::BOpt; kwargs...) = acquire_max(MaxMean(), o.model, o.lowerbounds, o.upperbounds; o.acquisitionoptions..., kwargs...)
# TODO: stochastic acquisition like Gutmann and Corander (2016), p. 20
function acquire_max(a::AbstractAcquisition, model, lowerbounds, upperbounds;
                     restarts = 1, kwargs...)
    opt = nlopt_setup(a, model, lowerbounds, upperbounds; kwargs...)
    acquire_max(opt, lowerbounds, upperbounds, restarts)
end

function acquire_max(opt, lowerbounds, upperbounds, restarts)
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


# TODO see
# https://github.com/HildoBijl/GPRT/blob/7166548b8587201fabc671a0647aac2ff96f3555/Chapter6/Chapter6.m#L723
# and corresponding thesis page 173
mutable struct ThompsonSampling{K} <: AbstractAcquisition
    np::Int64
    nc::Int64
    nr::Int64
    α::Float64
    kernel::K
end

# copied from https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/src/utilities/latin_hypercube_sampling.jl
function latin_hypercube_sampling(mins::AbstractVector{T},
                                  maxs::AbstractVector{T},
                                  n::Integer) where T<:Number
    length(mins) == length(maxs) ||
        throw(DimensionMismatch("mins and maxs should have the same length"))
    all(xy -> xy[1] <= xy[2], zip(mins, maxs)) ||
        throw(ArgumentError("mins[i] should not exceed maxs[i]"))
    dims = length(mins)
    result = zeros(T, dims, n)
    cubedim = Vector{T}(undef, n)
    @inbounds for i in 1:dims
        imin = mins[i]
        dimstep = (maxs[i] - imin) / n
        for j in 1:n
            cubedim[j] = imin + dimstep * (j - 1 + rand(T))
        end
        result[i, :] .= Random.shuffle!(cubedim)
    end
    result
end


function acquire_max(a::ThompsonSampling, model, lowerbounds, upperbounds; kwargs...)
    particles = [sample(lowerbounds, upperbounds) for _ in 1:a.np]
    weights = ones(a.np)
    for round in 1:a.nr
        round > 1 && resample!(particles, weigths)
        for i in 1:a.np

        end
    end
end
