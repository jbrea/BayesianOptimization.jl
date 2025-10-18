const GP = GaussianProcesses
function mean_var(model::GPBase, x::AbstractArray{<:Any, 1})
    μ, var = GP.predict_f(model, reshape(x, :, 1))
    μ[1], var[1]
end
myrand(model::GPBase, x::AbstractArray{<:Any, 1}) = rand(model, reshape(x, :, 1))[1]
myrand(model::GPBase, x) = rand(model, x)
mean_var(model::GPBase, x::AbstractArray{<:Any, 2}) = GP.predict_f(model, x)
dims(model::GPBase) = size(model.x)
maxy(model::GPBase) = length(model.y) == 0 ? -Inf : maximum(model.y)
update!(model::GPE{<:ElasticArray}, x, y) = append!(model, x, y)
function update!(model::GPE, x, y)
    if isempty(y)
        GP.fit!(model, model.x, model.y)
    else
        GP.fit!(model, hcat(model.x, x), [model.y; y])
    end
end

mutable struct MAPGPOptimizer{NT} <: ModelOptimizer
    i::Int
    every::Int
    options::NT
end
"""
    MAPGPOptimizer(; every = 10, kwargs...)

Set the GP hyperparameters to the maximum a posteriori (MAP) estimate `every`
number of steps. Run `BayesianOptimization.defaultoptions(MAPGPOptimizer)` to
see the default options. By default, all priors are flat. Uniform priors in an
interval can be specified by setting the bounds of the form
`[lowerbound, upperbound]`, e.g. for a kernel with 3 parameters one would set
`kernbounds = [[-3, -3, -4], [2, 3, 1]]`. Non-flat priors can be specified
directly on the GP parameters, e.g.
`using Distributions; set_priors!(mean, [Normal(0., 1.)])`
"""
function MAPGPOptimizer(; every = 10, kwargs...)
    MAPGPOptimizer(0, every,
                   merge(defaultoptions(MAPGPOptimizer),
                         kwargs)) #Changed kwargs.data -> kwargs
end
function optimizemodel!(o::MAPGPOptimizer, model::GPBase)
    if o.i % o.every == 0
        optimizemodel!(model, o.options)
    end
    o.i += 1
end
function defaultoptions(::Type{MAPGPOptimizer})
    (domean = true, kern = true, noise = true, lik = true, meanbounds = nothing,
     kernbounds = nothing, noisebounds = nothing, likbounds = nothing,
     method = :LD_LBFGS, maxeval = 500)
end

function optimizemodel!(gp::GPBase, options)
    params_kwargs = GP.get_params_kwargs(gp; domean = options.domean,
                                         kern = options.kern,
                                         noise = options.noise,
                                         lik = options.lik)
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
    fx, x, ret = NLopt.optimize(opt, GP.get_params(gp; params_kwargs...))
    ret == NLopt.FORCED_STOP && @warn("NLopt returned FORCED_STOP while optimizing the GP.")
    fx, x, ret
end
