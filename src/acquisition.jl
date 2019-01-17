### acquisition optimization setup

defaultoptions(::Type{<:GPE}, ::Type{<:AbstractAcquisition}) =
    (method = :LD_LBFGS, restarts = 10, maxeval = 2000)
defaultoptions(::Type{<:GPE}, ::Type{ThompsonSamplingSimple}) =
    (method = :GN_DIRECT_L, restarts = 1, maxeval = 2000)

function wrap_gradient(f)
    (x, g) -> begin
        res = DiffResults.DiffResult(0., g)
        ForwardDiff.gradient!(res, f, x)
        res.value
    end
end
wrap_dummygradient(f) = (x, g) -> f(x)

function nlopt_setup(a::AbstractAcquisition, model, lowerbounds, upperbounds,
                     options)
    D = length(lowerbounds)
    opt = NLopt.Opt(options.method, D)
    NLopt.maxeval!(opt, options.maxeval)
    NLopt.lower_bounds!(opt, lowerbounds)
    NLopt.upper_bounds!(opt, upperbounds)
    setparams!(a, model)
    if string(options.method)[2] == 'D'
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

acquire_max(o) = acquire_max(o.acquisition, o.model, o.lowerbounds, o.upperbounds, o.acquisitionoptions)
acquire_model_max(o; options = o.acquisitionoptions) = acquire_max(MaxMean(), o.model, o.lowerbounds, o.upperbounds, options)
# TODO: stochastic acquisition like Gutmann and Corander (2016), p. 20
function acquire_max(a::AbstractAcquisition, model, lowerbounds, upperbounds, options)
    opt = nlopt_setup(a, model, lowerbounds, upperbounds, options)
    acquire_max(opt, lowerbounds, upperbounds, options.restarts)
end

function acquire_max(opt, lowerbounds, upperbounds, restarts)
    maxf = -Inf
    maxx = lowerbounds
    x0s = latin_hypercube_sampling(lowerbounds, upperbounds, restarts)
    for i in 1:restarts
        f, x, ret = NLopt.optimize(opt, x0s[:, i])
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
function latin_hypercube_sampling(mins::AbstractVector,
                                  maxs::AbstractVector,
                                  n::Integer)
    length(mins) == length(maxs) ||
        throw(DimensionMismatch("mins and maxs should have the same length"))
    all(xy -> xy[1] <= xy[2], zip(mins, maxs)) ||
        throw(ArgumentError("mins[i] should not exceed maxs[i]"))
    dims = length(mins)
    result = zeros(dims, n)
    cubedim = Vector(undef, n)
    @inbounds for i in 1:dims
        imin = mins[i]
        dimstep = (maxs[i] - imin) / n
        for j in 1:n
            cubedim[j] = imin + dimstep * (j - 1 + rand())
        end
        result[i, :] .= Random.shuffle!(cubedim)
    end
    result
end
