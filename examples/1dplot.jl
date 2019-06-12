using BayesianOptimization, GaussianProcesses, PGFPlotsX, Random
push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{fillbetween}")

Random.seed!(134)
f(x, noisevariance = 1) = .1*sum((x .- 2).^2) + cos(sum(π/2 * x)) + noisevariance * randn()
model = ElasticGPE(1, mean = MeanConst(0.),
                   kernel = SEArd([0.], 5.), logNoise = 0.)
modeloptimizer = MAPGPOptimizer(every = 50, noisebounds = [-2., 3],
                                kernbounds = [[-1, 0], [4, 10]], maxeval = 40)
opt = BOpt(f, model, ExpectedImprovement(),
           modeloptimizer, [-5.], [5.],
           maxiterations = 5, sense = Min, repetitions = 5,
           acquisitionoptions = (maxeval = 4000, restarts = 50),
           verbosity = Progress)
result = boptimize!(opt)

acqfunc = BayesianOptimization.acquisitionfunction(opt.acquisition, model)
xs = -5:.02:5
ms, var = predict_f(model, xs)
sig = sqrt.(var)
fmax, xmax = BayesianOptimization.acquire_max(opt.opt,
                        opt.lowerbounds, opt.upperbounds,
                        opt.acquisitionoptions.restarts)
@pgf GroupPlot({group_style = {group_size = "1 by 2", vertical_sep = "4mm"},
                height = "4cm", width = "8cm", legend_pos = "outer north east",
                legend_style = {draw = "none"}},
               {legend_columns = 1, xticklabels = ""},
             Plot({only_marks}, Coordinates(model.x[:], -model.y[:])),
             "\\addlegendentry{observations}",
             Plot({no_marks, "red!20", name_path = "A", forget_plot}, Coordinates(xs, -ms .+ sig)),
             Plot({no_marks, "red!20", name_path = "B", forget_plot}, Coordinates(xs, -ms .- sig)),
             "\\addplot[red!20] fill between [of=A and B];",
             "\\addlegendentry{model std}",
             Plot({no_marks, blue, very_thick}, Coordinates(xs, -ms)),
             "\\addlegendentry{model mean}",
             Plot({no_marks, "green!80!black", very_thick}, Coordinates(xs, f.(xs, 0))),
             "\\addlegendentry{noisefree target}",
             {height = "3cm", ytick=[0, .05, .1], yticklabels = [0, "0.05", "0.1"]},
             Plot({only_marks, red, mark="triangle*",
                   mark_size = "5pt", mark_options = {rotate = "0"}},
                  Coordinates(xmax, [fmax])),
             Plot({no_marks, very_thick}, Coordinates(xs, (x -> acqfunc([x])).(xs))),
             "\\addlegendentry{next acquisition}",
             "\\addlegendentry{acquisition function}")
