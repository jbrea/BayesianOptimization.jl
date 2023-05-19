@testset "acquisition" begin
    model = GPE([1.0], [2.0], MeanZero(), SEIso(1.0, 0.0))
    ac = BO.MaxMean()
    opt = BO.nlopt_setup(ac, model, [-5.0], [5.0],
                         merge(BO.defaultoptions(typeof(model), typeof(ac)),
                               (maxtime = 3.0, ftol_abs = eps())))
    @test opt.maxeval == 2000
    @test opt.maxtime == 3.0
    @test opt.ftol_abs == eps()

    maxf, maxx = BO.acquire_max(opt, [-5.0], [5.0], 10)
    @test maxx ≈ [1.0]
end
