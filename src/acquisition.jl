abstract type AbstractAcquisition end

setparams!(a, model) = nothing

normal_pdf(μ, σ2) = 1/√(2π*σ2) * exp(-μ^2/(2*σ2))
normal_cdf(μ, σ2) = 1/2 * (1 + erf(μ/√(2σ2)))

"""
The probability of improvement measures the probability that a point `x` leads
to an improvement upon an incumbent target `τ`. For Gaussian distributions it is
given by

    Φ[(μ(x) - τ)/σ(x)]

where `Φ` is the standard normal cumulative distribution function and `μ(x)`, `σ(x)`
are mean and standard deviation of the distribution at point `x`.
"""
mutable struct ProbabilityOfImprovement <: AbstractAcquisition
    τ::Float64
end
function acquisitionfunction(a::ProbabilityOfImprovement, model)
    x -> begin
        μ, σ2 = mean_var(model, x)
        σ2 == 0 && return float(μ > a.τ)
        normal_cdf(μ - a.τ, σ2)
    end
end
ProbabilityOfImprovement(; τ = -Inf) = ProbabilityOfImprovement(τ)

"""
The expected improvement measures the expected improvement `x - τ` of a point `x`
upon an incumbent target `τ`. For Gaussian distributions it is given by

    (μ(x) - τ) * ϕ[(μ(x) - τ)/σ(x)] + σ(x) * Φ[(μ(x) - τ)/σ(x)]

where `ϕ` is the standard normal distribution function and `Φ` is the standard
normal cumulative function, and `μ(x)`, `σ(x)` are mean and standard deviation
of the distribution at point `x`.
"""
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

abstract type BetaScaling end
"""
Scales `βt` of `UpperConfidenceBound` as

    βt = √(2 * log(t^(D/2 + 2) * π^2/(3δ)))

where `t` is the number of observations, `D` is the dimensionality of the input
data points and δ is a small constant (default δ = 0.1).

See Brochu E., Cora V. M., de Freitas N. (2010), "A Tutorial on Bayesian
Optimization of Expensive Cost Functions, with Application to Active User
Modeling and Hierarchical Reinforcement Learning", https://arxiv.org/abs/1012.2599v1
page 16.
"""
struct BrochuBetaScaling <: BetaScaling
    δ::Float64
end
"""
Applies no scaling to `βt` of `UpperConfidenceBound`.
"""
struct NoBetaScaling <: BetaScaling end
"""
For Gaussian distributions the upper confidence bound at `x` is given by
    μ(x) + βt * σ(x)

where `βt` is a fixed parameter in the case of `NoBetaScaling` or an observation
size dependent parameter in the case of e.g. `BrochuBetaScaling`.
"""
mutable struct UpperConfidenceBound{S} <: AbstractAcquisition
    scaling::S
    βt::Float64
end
"""
    UpperConfidenceBound(; scaling = BrochuBetaScaling(.1), βt = 1)
"""
UpperConfidenceBound(; scaling = BrochuBetaScaling(.1), βt = 1.) = UpperConfidenceBound(scaling, βt)
function setparams!(a::UpperConfidenceBound{BrochuBetaScaling}, model)
    D, nobs = dims(model)
    nobs == 0 && (nobs = 1)
    a.βt = sqrt(2*log(nobs^(D/2 + 2)*π^2/(3*a.scaling.δ)))
end
function acquisitionfunction(a::UpperConfidenceBound, model)
    x -> begin
        μ, σ2 = mean_var(model, x)
        μ + a.βt * √σ2
    end
end

"""
The acquisition function associated with `ThompsonSamplingSimple` draws
independent samples for each input `x` a function value from the model. Together
with a gradient-free optimization method this leads to proposal points that
might be similarly distributed as the maxima of true Thompson samples from GPs.
True Thompson samples from a GP are simply functions from a GP. Maximizing these
samples can be tricky, see e.g. http://hildobijl.com/Downloads/GPRT.pdf
chapter 6.
"""
struct ThompsonSamplingSimple <: AbstractAcquisition end
acquisitionfunction(a::ThompsonSamplingSimple, model) = x -> rand(model, reshape(x, :, 1))[1]

struct MaxMean <: AbstractAcquisition end
acquisitionfunction(a::MaxMean, model) = x -> mean_var(model, x)[1]

# TODO see
# https://github.com/HildoBijl/GPRT/blob/7166548b8587201fabc671a0647aac2ff96f3555/Chapter6/Chapter6.m#L723
# and corresponding thesis page 173
# mutable struct ThompsonSampling{K} <: AbstractAcquisition
#     np::Int
#     nc::Int
#     nr::Int
#     α::Float64
#     kernel::K
# end
#
# function acquire_max(a::ThompsonSampling, model, lowerbounds, upperbounds; kwargs...)
#     particles = [sample(lowerbounds, upperbounds) for _ in 1:a.np]
#     weights = ones(a.np)
#     for round in 1:a.nr
#         round > 1 && resample!(particles, weigths)
#         for i in 1:a.np
#
#         end
#     end
# end
#
# TODO
# mutable struct EntropySearch <: AbstractAcquisition
# end

# TODO
# mutable struct PredictiveEntropySearch <: AbstractAcquisition
# end

"""
The mutual information measures the amount of information gained by querying at
x. The parameter γ̂ gives a lower bound for the information on f from the queries
{x}. For a Guassian this is
    γ̂ = ∑σ²(x)
and the mutual information at x is
    μ(x) + √(α)*(√(σ²(x)*γ̂) - √(γ̂))

where `μ(x)`, `σ(x)` are mean and standard deviation
of the distribution at point `x`.

See Contal E., Perchet V., Vayatis N. (2014), "Gaussian Process Optimization
with Mutual Information" http://proceedings.mlr.press/v32/contal14.pdf
"""
mutable struct MutualInformation <: AbstractAcquisition
    α::Float64
    γ̂::Float64
end
MutualInformation(; α = 1.0, γ̂ = 0.0) = MutualInformation(α, γ̂)
function setparams!(a::MutualInformation, model)
    D, nobs = dims(model)
    if iszero(nobs)
        a.γ̂ = 0.0
    else
        last_x = model.x[:, end]
        μ, σ2 = mean_var(model, last_x)
        a.γ̂ += σ2
    end
end
function acquisitionfunction(a::MutualInformation, model)
    x -> begin
        μ, σ2 = mean_var(model, x)
        μ + sqrt(a.α) * (sqrt(σ2 + a.γ̂) - sqrt(a.γ̂))
    end
end

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
        ret == NLopt.FORCED_STOP && throw(InterruptException())
        if f > maxf
            maxf = f
            maxx = x
        end
    end
    maxf, maxx
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
