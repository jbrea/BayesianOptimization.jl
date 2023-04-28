
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
    for (option, value) in pairs(options)
        (option == :method || option == :restarts) && continue
        setproperty!(opt, option, value)
    end
    NLopt.lower_bounds!(opt, lowerbounds)
    NLopt.upper_bounds!(opt, upperbounds)
    setparams!(a, model)
    if string(options.method)[2] == 'D'
        f = wrap_gradient(acquisitionfunction(a, model))
        NLopt.max_objective!(opt, f)
    else
        NLopt.max_objective!(opt, wrap_dummygradient(acquisitionfunction(a, model)))
    end
    opt
end


acquire_max(o) = acquire_max(o.acquisition, o.model, o.lowerbounds, o.upperbounds, o.acquisitionoptions)

acquire_model_max(o; options = o.acquisitionoptions) = acquire_max(MaxMean(), o.model, o.lowerbounds, o.upperbounds, options)
function acquire_max(a::AbstractAcquisition, model, lowerbounds, upperbounds, options)
    opt = nlopt_setup(a, model, lowerbounds, upperbounds, options)
    acquire_max(opt, lowerbounds, upperbounds, options.restarts)
end

function acquire_max(opt, lowerbounds, upperbounds, restarts)
    maxf = -Inf
    maxx = lowerbounds
    seq = ScaledLHSIterator(lowerbounds, upperbounds, restarts)
    for x0 in seq
        f, x, ret = NLopt.optimize(opt, x0)
        ret == NLopt.FORCED_STOP && @warn("NLopt returned FORCED_STOP while optimizing the acquisition function.")
        if f > maxf
            maxf = f
            maxx = x
        end
    end
    maxf, maxx
end
