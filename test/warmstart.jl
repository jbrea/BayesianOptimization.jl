ac = ExpectedImprovement()
x_premade = rand(2, 10) * 15.0 .- [5.0; 0.0]
y_premade = -1 * [branin(x_premade[:, i], noiselevel = 0) for i in 1:size(x_premade, 2)]

model_premade = ElasticGPE(2, mean = MeanConst(-10.0),
                           kernel = SEArd([0.0, 0.0], 5.0), logNoise = -2.0)
append!(model_premade, x_premade, y_premade)

# (#7)
@testset "Initial Sampling Tracking" begin
    opt = BOpt(x -> branin(x, noiselevel = 0),
               ElasticGPE(2, mean = MeanConst(-10.0),
                          kernel = SEArd([0.0, 0.0], 5.0),
                          logNoise = -2.0, capacity = 3000),
               ac,
               MAPGPOptimizer(every = 50, noisebounds = [-4, 3],
                              kernbounds = [[-1, -1, 0], [4, 4, 10]],
                              maxeval = 40),
               [-5.0, 0.0], [10.0, 15.0],
               maxiterations = 10,
               sense = Min,
               verbosity = Silent,
               initializer_iterations = 10)
    boptimize!(opt)

    @test opt.observed_optimum == Int(opt.sense) * maximum(opt.model.y)
    @test length(opt.model.y) == 10
end

# (#7)
@testset "Pre-made model" begin
    opt = BOpt(x -> branin(x, noiselevel = 0),
               model_premade,
               ac,
               MAPGPOptimizer(every = 50, noisebounds = [-4, 3],
                              kernbounds = [[-1, -1, 0], [4, 4, 10]],
                              maxeval = 40),
               [-5.0, 0.0], [10.0, 15.0],
               maxiterations = 10,
               sense = Min,
               verbosity = Silent,
               initializer_iterations = 5)

    @test opt.observed_optimum == Int(opt.sense) * maximum(y_premade)
    @test opt.observed_optimizer == x_premade[:, argmax(y_premade)]
end

# (#7)
@testset "Initial iterations to 0 on pre-made model" begin
    ac = ExpectedImprovement()
    opt = BOpt(x -> branin(x, noiselevel = 0),
               model_premade,
               ac,
               MAPGPOptimizer(every = 50, noisebounds = [-4, 3],
                              kernbounds = [[-1, -1, 0], [4, 4, 10]],
                              maxeval = 40),
               [-5.0, 0.0], [10.0, 15.0],
               maxiterations = 0,
               sense = Min,
               verbosity = Silent,
               initializer_iterations = 0)
    boptimize!(opt)

    @test opt.acquisition.τ == maximum(y_premade)
    @test length(opt.model.x) == length(x_premade)
    @test length(opt.model.y) == length(y_premade)

    opt.iterations.N = 5
    boptimize!(opt)
    @test length(opt.model.y) == length(y_premade) + 5
end
