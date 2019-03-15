@testset "acquisition functions" begin
    for ac in [ProbabilityOfImprovement(), ExpectedImprovement(),
               UpperConfidenceBound(), ThompsonSamplingSimple(), MutualInformation()]
        model = GPE(rand(3, 4), rand(4), MeanZero(), SEIso(0., 0.))
        acfunc = BayesianOptimization.acquisitionfunction(ac, model)
        x = rand(3, 2)
        acvector = acfunc(x)
        @test length(acvector) == 2
        if typeof(ac) != ThompsonSamplingSimple
            @test acvector[1] == acfunc(x[:, 1])
        end
    end
end
