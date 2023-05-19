@testset "utils" begin
    import BayesianOptimization: IterationCounter, DurationCounter, step!,
                                 init!, isdone
    it = IterationCounter(0, 0, 10)
    for _ in 1:10
        step!(it)
    end
    @test isdone(it) == true
    maxiterations!(it, 40)
    @test it.N == 40
    init!(it)
    @test it.c == 0
    @test it.i == 10

    now = time()
    it = DurationCounter(now, 0.2, now, now + 0.2)
    sleep(0.21)
    @test isdone(it) == true
    maxduration!(it, 0.5)
    init!(it)
    @test isdone(it) == false
    sleep(0.51)
    @test isdone(it) == true
end
