function mean_var(model::GPBase, x::AbstractArray{<:Any, 1})
    μ, var = GaussianProcesses.predict_f(model, reshape(x, :, 1))
    μ[1], var[1]
end
dims(model::GPBase) = size(model.x)
maxy(model::GPBase) = length(model.y) == 0 ? -Inf : maximum(model.y)
update!(model::GPE{X,Y,M,K,P,D}, x, y) where {X,Y,M,K,P<:ElasticPDMat, D} = append!(model, x, y)
function update!(model::GPE{X,Y,M,K,P,D}, x, y) where {X,Y,M,K,P,D}
    GaussianProcesses.fit!(model, hcat(model.x, x), [model.y; y])
end

