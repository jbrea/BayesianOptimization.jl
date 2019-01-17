macro mytimeit(exprs...)
    if ENABLE_TIMINGS
        return :(@timeit($(esc.(exprs)...)))
    else
        return esc(exprs[end])
    end
end

mutable struct IterationCounter
    c::Int
    i::Int
    N::Int
end
isdone(s::IterationCounter) = s.c == s.N
step!(s::IterationCounter) = (s.c += 1; s.i += 1)
init!(s::IterationCounter) = s.c = 0

mutable struct DurationCounter
    starttime::Float64
    duration::Float64
    now::Float64
    endtime::Float64
end
function init!(s::DurationCounter)
    s.starttime = time()
    s.endtime = s.starttime + s.duration
end
isdone(s::DurationCounter) = (s.now = time()) >= s.endtime

sample(lowerbounds, upperbounds) =
    rand(length(lowerbounds)) .* (upperbounds .- lowerbounds) .+ lowerbounds

normal_pdf(μ, σ2) = 1/√(2π*σ2) * exp(-μ^2/(2*σ2))
normal_cdf(μ, σ2) = 1/2 * (1 + erf(μ/√(2σ2)))
