using BayesianOptimization, GaussianProcesses, Random, Statistics

# function and regret definitions
branin(x::Vector; kwargs...) = branin(x[1], x[2]; kwargs...)
function branin(x1, x2; a = 1, b = 5.1 / (4π^2), c = 5 / π, r = 6, s = 10, t = 1 / (8π),
                noiselevel = 0)
    a * (x2 - b * x1^2 + c * x1 - r)^2 + s * (1 - t) * cos(x1) + s + noiselevel * randn()
end

minima(::typeof(branin)) = [[-π, 12.275], [π, 2.275], [9.42478, 2.475]], 0.397887

function hartmann(x; α = [1.0, 1.2, 3.0, 3.2],
                  A = [10 3 17 3.5 1.7 8;
                       0.05 10 17 0.1 8 14;
                       3 3.5 1.7 10 17 8;
                       17 8 0.05 10 0.1 14],
                  P = 1e-4 * [1312 1696 5569 124 8283 5886;
                       2329 4135 8307 3736 1004 9991;
                       2348 1451 3522 2883 3047 6650;
                       4047 8828 8732 5743 1091 381])
    -sum([α[i] * exp(-sum([A[i, j] * (x[j] - P[i, j])^2 for j in 1:6])) for i in 1:4])
end

function minima(::typeof(hartmann))
    [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]], -3.32237
end

euclidean(x, y) = √sum((x .- y) .^ 2)
function regret(opt, func)
    mins, fmin = minima(func)
    (observed_dist = minimum(map(m -> euclidean(m, opt.observed_optimizer), mins)),
     observed_regret = abs(opt.observed_optimum - fmin),
     model_dist = minimum(map(m -> euclidean(m, opt.model_optimizer), mins)),
     model_regret = abs(Int(opt.sense) * opt.model_optimum - fmin))
end

# optimize noise-free branin

opt = BOpt(x -> branin(x, noiselevel = 0),
           ElasticGPE(2, mean = MeanConst(-10.0), kernel = SEArd([0.0, 0.0], 5.0),
                      logNoise = -2.0, capacity = 3000),
           ExpectedImprovement(),
           MAPGPOptimizer(every = 50, noisebounds = [-4, 3],
                          kernbounds = [[-1, -1, 0], [4, 4, 10]],
                          maxeval = 40),
           [-5.0, 0.0], [10.0, 15.0], maxiterations = 200,
           sense = Min)
@time boptimize!(opt)
regret(opt, branin)

# optimize hartman

opt = BOpt(hartmann,
           ElasticGPE(6, mean = MeanConst(0.0), kernel = Mat52Ard(zeros(6), 0.0),
                      logNoise = -2.0, capacity = 3000),
           ExpectedImprovement(),
           MAPGPOptimizer(every = 20, noisebounds = [-4, 3],
                          kernbounds = [[-3 * ones(6); -3], [4 * ones(6); 3]],
                          maxeval = 100),
           zeros(6), ones(6), maxiterations = 300,
           sense = Min)
@time boptimize!(opt)
regret(opt, hartmann)

# compare model and observation optimizer on noisy branin

all_obs_regs = []
all_mod_regs = []
for _ in 1:10
    opt = BOpt(x -> branin(x, noiselevel = 1),
               ElasticGPE(2, mean = MeanConst(-50.0), kernel = SEArd(zeros(2), 4.0),
                          logNoise = 2.0, capacity = 1000),
               UpperConfidenceBound(),
               MAPGPOptimizer(every = 100, noisebounds = [-4, 3],
                              kernbounds = [[-1, -1, 0], [4, 4, 10]],
                              f_calls_limit = 40), repetitions = 5,
               [-5.0, 0.0], [10.0, 15.0], maxiterations = 20,
               sense = Min, verbosity = Silent)
    obs_regs = []
    mod_regs = []
    for i in 1:10
        res = boptimize!(opt)
        _, obs_reg, _, mod_reg = regret(opt, branin)
        push!(obs_regs, obs_reg)
        push!(mod_regs, mod_reg)
    end
    push!(all_obs_regs, obs_regs)
    push!(all_mod_regs, mod_regs)
end
using PGFPlotsX
x = (1:length(all_obs_regs[1])) * 20 * 5
@pgf Axis({ymode = "log",
           legend_entries = ["average observation regret",
               "average model regret"],
           legend_columns = 1, legend_pos = "north east",
           ylabel = "number of observations",
           title = "model optimizers become more accurate for noisy objectives"},
          Plot({red, very_thick, no_marks}, Coordinates(x, mean(all_obs_regs))),
          Plot({blue, very_thick, no_marks}, Coordinates(x, mean(all_mod_regs))),
          [Plot({red, thin, forget_plot}, Coordinates(x, y)) for y in all_obs_regs]...,
          [Plot({blue, thin, forget_plot}, Coordinates(x, y)) for y in all_mod_regs]...)
