@testset "Merge with defaults" begin
    f = x -> x[1]+x[2]
    l = [-1,-2]
    u = [3, 4]
    
    args, kwargs = BO.merge_with_defaults(f,l,u, (sense=Max, acquisition=ProbabilityOfImprovement(), maxiterations=20))

    @test length(args) == 6
    @test args[1] == f
    @test isa(args[3], ProbabilityOfImprovement)
    @test kwargs.sense == Max
    @test kwargs.maxiterations == 20
    # overwriting f, lowerbounds, upperbounds in optkwargs is not permitted, test if unsupported kwargs result in ArgumentError
    @test_throws ArgumentError BO.merge_with_defaults(f,l,u, (;func=(x -> x[1])))
    @test_throws ArgumentError BO.merge_with_defaults(f,l,u, (lowerbounds=[],upperbounds=[-1,-100]))
    @test_throws ArgumentError BO.merge_with_defaults(f,l,u, (;hello="world"))
    # length of lowerbound not eq. to lenght of upperbound results in an ArgumentError
    @test_throws ArgumentError BO.merge_with_defaults(f,[1.],[3.,4.], (;))
    # wrong order of lowerbound, upperbound args results in an ArgumentError
    @test_throws ArgumentError BO.merge_with_defaults(f,u,l,(;))
end