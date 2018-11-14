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

function optimizemodel!(gp::GPBase; 
                        domean = true, kern = true, noise = true, lik = true,
                        meanbounds = nothing, kernbounds = nothing, 
                        noisebounds = nothing, likbounds = nothing, 
                        method = :LD_LBFGS, 
                        maxeval = 500)
    params_kwargs = GP.get_params_kwargs(gp; domean=domean, kern=kern, 
                                         noise=noise, lik=lik)
    f = (x, g) -> begin
            GP.set_params!(gp, x; params_kwargs...)
            GP.update_target_and_dtarget!(gp; params_kwargs...)
            @. g = gp.dtarget
            gp.target
        end
    lb, ub = GP.bounds(gp, noisebounds, meanbounds, kernbounds, likbounds;
                       domean = domean, kern = kern, noise = noise, lik = lik)
    opt = NLopt.Opt(method, length(lb))
    NLopt.lower_bounds!(opt, lb)
    NLopt.upper_bounds!(opt, ub)
    NLopt.maxeval!(opt, maxeval)
    NLopt.max_objective!(opt, f)
    f, x, ret = NLopt.optimize(opt, GP.get_params(gp; params_kwargs...))
    ret == NLopt.FORCED_STOP && throw(InterruptException())
    f, x, ret
end

