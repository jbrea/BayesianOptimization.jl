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
"""
     maxiterations!(s::IterationCounter, N)
     maxiterations!(o::BOpt, N)

Sets the maximal number of iterations per call of `boptimize!` to `N`.
"""
maxiterations!(s::IterationCounter, N) = s.N = N

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
"""
     maxduration!(s::IterationCounter, duration)
     maxduration!(o::BOpt, duration)

Sets the maximal duration per call of `boptimize!` to `duration`.
"""
maxduration!(s::DurationCounter, d) = s.duration = d

sample(lowerbounds, upperbounds) =
    rand(length(lowerbounds)) .* (upperbounds .- lowerbounds) .+ lowerbounds

normal_pdf(μ, σ2) = 1/√(2π*σ2) * exp(-μ^2/(2*σ2))
normal_cdf(μ, σ2) = 1/2 * (1 + erf(μ/√(2σ2)))

struct ColumnIterator{T <: AbstractMatrix, Tit}
    data::T
    baseiterator::Tit
end
ColumnIterator(data::AbstractMatrix) = ColumnIterator(data, 1:size(data, 2))
import Base: iterate, length
@inline function dispatch(it::ColumnIterator, next)
    next === nothing && return nothing
    @view(it.data[:, next[1]]), next[2]
end
iterate(it::ColumnIterator) = dispatch(it, iterate(it.baseiterator))
iterate(it::ColumnIterator, s) = dispatch(it, iterate(it.baseiterator, s))
length(it::ColumnIterator) = length(it.baseiterator)
struct ScaledSobolIterator{T,D}
    lowerbounds::Vector{T}
    upperbounds::Vector{T}
    N::Int
    seq::SobolSeq{D}
end
"""
    ScaledSobolIterator(lowerbounds, upperbounds, N;
                        seq = SobolSeq(length(lowerbounds)))

Returns an iterator over `N` elements of a Sobol sequence between `lowerbounds`
and `upperbounds`. The first `N` elements of the Sobol sequence are skipped for
better uniformity (see https://github.com/stevengj/Sobol.jl)
"""
function ScaledSobolIterator(lowerbounds, upperbounds, N;
                             seq = SobolSeq(length(lowerbounds)))
    N > 0 && skip(seq, N)
    ScaledSobolIterator(lowerbounds, upperbounds, N, seq)
end
length(it::ScaledSobolIterator) = it.N
@inline function iterate(it::ScaledSobolIterator, s = 1)
    s == it.N + 1 && return nothing
    Sobol.next!(it.seq, it.lowerbounds, it.upperbounds), s + 1
end

"""
     ScaledLHSIterator(lowerbounds, upperbounds, N)

Returns an iterator over `N` elements of a latin hyper cube sample between
`lowerbounds` and `upperbounds`. See also `ScaledSobolIterator` for an iterator
that has arguably better uniformity.
"""
function ScaledLHSIterator(lowerbounds, upperbounds, N)
    ColumnIterator(latin_hypercube_sampling(lowerbounds, upperbounds, N))
end

# copied from https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/src/utilities/latin_hypercube_sampling.jl
function latin_hypercube_sampling(mins::AbstractVector,
                                  maxs::AbstractVector,
                                  n::Integer)
    length(mins) == length(maxs) ||
        throw(DimensionMismatch("mins and maxs should have the same length"))
    all(xy -> xy[1] <= xy[2], zip(mins, maxs)) ||
        throw(ArgumentError("mins[i] should not exceed maxs[i]"))
    dims = length(mins)
    result = zeros(dims, n)
    cubedim = Vector(undef, n)
    @inbounds for i in 1:dims
        imin = mins[i]
        dimstep = (maxs[i] - imin) / n
        for j in 1:n
            cubedim[j] = imin + dimstep * (j - 1 + rand())
        end
        result[i, :] .= Random.shuffle!(cubedim)
    end
    result
end
