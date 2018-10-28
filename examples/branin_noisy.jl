branin(x::Vector; kwargs...) = branin(x[1], x[2]; kwargs...)
branin(x1, x2; a = 1, b = 5.1/(4π^2), c = 5/π, r = 6, s = 10, t = 1/(8π), 
       noiselevel = 0) = 
    a * (x2 - b*x1^2 + c*x1 - r)^2 + s*(1 - t)*cos(x1) + s + noiselevel * randn()

minima(::typeof(branin)) = [([-π, 12.275], 0.397887), ([π, 2.275], 0.397887), ([9.42478, 2.475], 0.397887)]

function regret(opt, func)
    mins = [m[1] for m in minima(func)]
    (observed_regret = minimum(map(m -> euclidean(m, opt.optimizer), mins)),
     model_regret = minimum(map(m -> euclidean(m, opt.model_optimizer), mins)))
end

using BayesianOptimization, GaussianProcesses, Distances

opt = BOpt(x -> branin(x, noiselevel = 0), 
           ElasticGPE(2, mean = MeanConst(-10.), kernel = SEArd([3., 3.], 5.),
                      logNoise = -2., capacity = 3000),
           ExpectedImprovement(), 
           GPOptimizer(every = 50, noise = false),
#            NoOptimizer(),
           [-5., 0.], [10., 15.], maxiterations = 100, 
           sense = Min)
@time BayesianOptimization.optimize!(opt)
regret(opt, branin)

all_obs_regs = []
all_mod_regs = []
for _ in 1:20
    opt = BOpt(x -> branin(x, noiselevel = 20), 
               ElasticGPE(2, mean = MeanConst(-50.), kernel = SEArd(rand(2), 4.),
                          logNoise = 2., capacity = 3000),
               UpperConfidenceBound(), 
               GPOptimizer(every = 1000),
               [-5., 0.], [10., 15.], maxiterations = 30, 
               sense = Min, 
               acquisitionoptions = Dict(:method => :LD_LBFGS, :maxeval => 1000))

    obs_regs = []
    mod_regs = []
    for i in 1:99
        res = BayesianOptimization.optimize!(opt);
        obs_reg, mod_reg = regret(opt, branin)
        push!(obs_regs, obs_reg)
        push!(mod_regs, mod_reg)
    end
    push!(all_obs_regs, obs_regs)
    push!(all_mod_regs, mod_regs)
end

using Statistics: mean
using Plots
p = plot(mean(all_obs_regs))
plot!(p, mean(all_mod_regs))

function test()
    model = PlasticCachingModel
    params_not_shared_by_birds = [:motivationrate]
    share_params_across_exp = true
    fp = FitParameters(model, share_params_across_exp = share_params_across_exp,
                       params_not_shared_by_birds = params_not_shared_by_birds,
                       experiments = Dict(:Cheke11_planning =>
                                           EXPERIMENTS[:Cheke11_planning]))
    bounds = map(k -> PlanningJays.SEARCHRANGE[k], fp.names)
    func = x -> -modelscore(getmodelsfunc(fp, x), experiments =
                            [:Cheke11_planning], Nsamples = 10)
    func, [b[1] for b in bounds], [b[2] for b in bounds]
end
f, lowerbounds, upperbounds = test()
opt = BOpt(f, 
           ElasticGPE(19, mean = MeanConst(.05), kernel = SEArd((upperbounds - lowerbounds)/10, 2.),
                      logNoise = -1., capacity = 3000),
           ExpectedImprovement(), 
           GPOptimizer(every = 10000),
           lowerbounds, upperbounds, maxiterations = 1000, 
           sense = Max, 
           acquisitionoptions = Dict(:method => :LD_LBFGS, :maxeval => 1000))

BayesianOptimization.optimize!(opt)

