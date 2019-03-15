branin(x::Vector; kwargs...) = branin(x[1], x[2]; kwargs...)
branin(x1, x2; a = 1, b = 5.1/(4π^2), c = 5/π, r = 6, s = 10, t = 1/(8π),
       noiselevel = 0) =
    a * (x2 - b*x1^2 + c*x1 - r)^2 + s*(1 - t)*cos(x1) + s + noiselevel * randn()

minima(::typeof(branin)) = [[-π, 12.275], [π, 2.275], [9.42478, 2.475]], 0.397887

euclidean(x, y) = √sum((x .- y).^2)
function regret(opt, func)
    mins, fmin = minima(func)
    (observed_dist = minimum(map(m -> euclidean(m, opt.observed_optimizer), mins)),
     observed_regret = abs(opt.observed_optimum - fmin),
     model_dist = minimum(map(m -> euclidean(m, opt.model_optimizer), mins)),
     model_regret = abs(opt.model_optimum - fmin))
end

@testset "branin" begin
    for ac in [ProbabilityOfImprovement(), ExpectedImprovement(),
               UpperConfidenceBound(), ThompsonSamplingSimple(), MutualInformation()]
        println("testing on branin with $ac")
        opt = BOpt(x -> branin(x, noiselevel = 0),
                   ElasticGPE(2, mean = MeanConst(-10.),
                              kernel = SEArd([0., 0.], 5.),
                              logNoise = -2., capacity = 3000),
                   ac,
                   MAPGPOptimizer(every = 50, noisebounds = [-4, 3],
                                 kernbounds = [[-1, -1, 0], [4, 4, 10]],
                                 maxeval = 40),
                   [-5., 0.], [10., 15.], maxiterations = 200,
                   sense = Min,
                   verbosity = Silent)
        boptimize!(opt)
        reg = regret(opt, branin)
        @test reg.observed_regret < 0.05
    end
end
