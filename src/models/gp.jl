const GP = GaussianProcesses
function mean_var(model::GPBase, x::AbstractArray{<:Any, 1})
    μ, var = GP.predict_f(model, reshape(x, :, 1))
    μ[1], var[1]
end
dims(model::GPBase) = size(model.x)
maxy(model::GPBase) = length(model.y) == 0 ? -Inf : maximum(model.y)
update!(model::GPE{X,Y,M,K,P,D}, x, y) where {X,Y,M,K,P<:ElasticPDMat, D} = append!(model, x, y)
function update!(model::GPE{X,Y,M,K,P,D}, x, y) where {X,Y,M,K,P,D}
    GP.fit!(model, hcat(model.x, x), [model.y; y])
end

mutable struct MLGPOptimizer{NT} <: ModelOptimizer
    i::Int
    every::Int
    options::NT
end
"""
    MLGPOptimizer(; every = 10, kwargs...)

Set the GP hyperparameters to the maximum likelihood estimate `every` number of steps.
"""
MLGPOptimizer(; every = 10, kwargs...) = MLGPOptimizer(0, every,
                                                       merge(defaultoptions(MLGPOptimizer),
                                                             kwargs.data))
function optimizemodel!(o::MLGPOptimizer, model::GPBase)
    if o.i % o.every == 0
        optimizemodel!(model, o.options)
    end
    o.i += 1
end
defaultoptions(::Type{MLGPOptimizer}) =
    (domean = true, kern = true, noise = true, lik = true, meanbounds = nothing,
     kernbounds = nothing, noisebounds = nothing, likbounds = nothing,
     method = :LD_LBFGS, maxeval = 500)

function optimizemodel!(gp::GPBase, options)
    params_kwargs = GP.get_params_kwargs(gp; domean=options.domean,
                                         kern=options.kern,
                                         noise=options.noise,
                                         lik=options.lik)
    f = (x, g) -> begin
            GP.set_params!(gp, x; params_kwargs...)
            GP.update_target_and_dtarget!(gp; params_kwargs...)
            @. g = gp.dtarget
            gp.target
        end
    lb, ub = GP.bounds(gp, options.noisebounds, options.meanbounds,
                       options.kernbounds, options.likbounds;
                       domean = options.domean, kern = options.kern,
                       noise = options.noise, lik = options.lik)
    opt = NLopt.Opt(options.method, length(lb))
    NLopt.lower_bounds!(opt, lb)
    NLopt.upper_bounds!(opt, ub)
    NLopt.maxeval!(opt, options.maxeval)
    NLopt.max_objective!(opt, f)
    f, x, ret = NLopt.optimize(opt, GP.get_params(gp; params_kwargs...))
    ret == NLopt.FORCED_STOP && throw(InterruptException())
    f, x, ret
end

