branin(x::Vector; kwargs...) = branin(x[1], x[2]; kwargs...)
branin(x1, x2; a = 1, b = 5.1/(4π^2), c = 5/π, r = 6, s = 10, t = 1/(8π), 
       noiselevel = 0) = 
    a * (x2 - b*x1^2 + c*x1 - r)^2 + s*(1 - t)*cos(x1) + s + noiselevel * randn()

minima(::typeof(branin)) = [([-π, 12.275], 0.397887), ([π, 2.275], 0.397887), ([9.42478, 2.475], 0.397887)]

using BayesianOptimization, GaussianProcesses
opt = BOpt(x -> branin(x, noiselevel = 10), GPE(Array{Float64}(undef, 2, 0), Float64[],
                                               MeanConst(-50.), 
                                               Mat52Ard([1., 2.], 1.), 2.,
                                               false, # true is more efficient and works with https://github.com/STOR-i/GaussianProcesses.jl/pull/93
                                              ),
           UpperConfidenceBound(), 
#            GPOptimizer(every = 500, noise = false),
           NoOptimizer(),
           [-5., 0.], [10., 15.], maxiterations = 1000, sense = Min)

res = BayesianOptimization.optimize!(opt)
opt
