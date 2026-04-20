abstract type AbstractIndexList <: AbstractVector{Index} end

ind_vect(inds::Vector{Index}) = inds
Base.@nospecializeinfer ind_vect(@nospecialize(inds::AbstractVector{<:Index})) = Vector{Index}(inds)
Base.@nospecializeinfer ind_vect(@nospecialize(inds::Tuple)) = ind_vect(inds...)

Base.@nospecializeinfer function ind_vect(@nospecialize(inds::Vararg{<:Index}))
    vec = Vector{Index}(undef, length(inds))
    for i in eachindex(inds)
        @inbounds vec[i] = inds[i]
    end
    return vec
end

function ind_vect(inds::Vector{Symbol})
    vec = Vector{Index}(undef, length(inds))
    for i in eachindex(inds)
        @inbounds vec[i] = Index(inds[i])
    end
    return vec
end

struct IndexList <: AbstractIndexList
    vec::Vector{Index}

    IndexList(vec::Vector{Index}) = new(vec)
    IndexList(itr) = new(ind_vect(itr))
end

IndexList() = IndexList(Vector{Index}())
IndexList(x::IndexList) = x
IndexList(::UndefInitializer, n::Int) = IndexList(Vector{Index}(undef, n))

struct MutableIndexList <: AbstractIndexList
    vec::Vector{Index}

    MutableIndexList(vec::Vector{Index}) = new(vec)
    MutableIndexList(itr) = new(ind_vect(itr))
end

MutableIndexList() = MutableIndexList(Vector{Index}())
MutableIndexList(x::MutableIndexList) = x
MutableIndexList(::UndefInitializer, n::Int) = MutableIndexList(Vector{Index}(undef, n))

IndexList(x::MutableIndexList) = IndexList(parent(x))

Base.parent(a::AbstractIndexList) = a.vec

Base.iterate(a::AbstractIndexList) = iterate(a.vec)
Base.iterate(a::AbstractIndexList, state) = iterate(a.vec, state)
Base.IteratorSize(::Type{<:AbstractIndexList}) = Base.HasShape{1}()
Base.length(a::AbstractIndexList) = length(a.vec)
Base.size(a::AbstractIndexList) = size(a.vec)
Base.size(a::AbstractIndexList, d::Int) = size(a.vec, d)
Base.IteratorEltype(::Type{<:AbstractIndexList}) = Base.HasEltype()

Base.getindex(a::AbstractIndexList, i::Int) = getindex(a.vec, i)
Base.getindex(a::T, r::AbstractRange{Int}) where {T<:AbstractIndexList} = T(getindex(a.vec, r))
Base.setindex!(a::MutableIndexList, v, i::Int) = setindex!(a.vec, v, i)
Base.setindex!(a::MutableIndexList, v, r::AbstractRange{Int}) = setindex!(a.vec, v, r)
Base.firstindex(a::AbstractIndexList) = firstindex(a.vec)
Base.lastindex(a::AbstractIndexList) = lastindex(a.vec)

Base.IndexStyle(::Type{<:AbstractIndexList}) = Base.IndexLinear()
Base.similar(::AbstractIndexList, ::Type{Index}, dims::Base.Dims{1}) = MutableIndexList(undef, dims[1])
function Base.similar(::AbstractIndexList, ::Type{T}, dims::Base.Dims{1}) where {T}
    @debug "[Base.similar] creating Vector{$T} instead of MutableIndexList"
    return MutableIndexList(undef, dims[1])
end

Base.push!(a::MutableIndexList, v) = push!(a.vec, v)
Base.pop!(a::MutableIndexList) = pop!(a.vec)
Base.insert!(a::MutableIndexList, i::Int, v) = insert!(a.vec, i, v)
Base.deleteat!(a::MutableIndexList, i::Int) = deleteat!(a.vec, i)
Base.append!(a::MutableIndexList, v) = append!(a.vec, v)
Base.empty!(a::MutableIndexList) = empty!(a.vec)

# used by Base.union
Base.emptymutable(::AbstractIndexList, ::Type{Index}=Index) = MutableIndexList()
function Base.emptymutable(::AbstractIndexList, ::Type{T}) where {T}
    @debug "[Base.emptymutable] creating Vector{$T} instead of MutableIndexList"
    return Vector{T}()
end

Base.intersect(a::AbstractIndexList, b) = IndexList(intersect(parent(a), b))
Base.intersect!(a::MutableIndexList, b) = MutableIndexList(intersect!(parent(a), b))
Base.setdiff(a::AbstractIndexList, b) = IndexList(setdiff(parent(a), b))
Base.setdiff!(a::MutableIndexList, b) = MutableIndexList(setdiff!(parent(a), b))
Base.symdiff(a::AbstractIndexList, b) = IndexList(symdiff(parent(a), b))
Base.symdiff!(a::MutableIndexList, b) = MutableIndexList(symdiff!(parent(a), b))

function findperm(from::AbstractIndexList, to::AbstractIndexList)
    @assert issetequal(from, to)

    # if there are hyperindices, we remove one by one
    inds_to = collect(Union{Missing,Index}, to)
    res = Vector{Int}(undef, length(from))

    for i in eachindex(from)
        j = findfirst(isequal(from[i]), inds_to)

        # mark element as used
        inds_to[j] = missing

        res[i] = j
    end

    return res
end

function factorinds(_all_inds, _left_inds, _right_inds)
    all_inds = IndexList(_all_inds)
    left_inds = IndexList(_left_inds)
    right_inds = IndexList(_right_inds)

    if !isdisjoint(left_inds, right_inds)
        throw(ArgumentError("left ($left_inds) and right $(right_inds) indices must be disjoint"))
    end

    left_inds, right_inds = if isempty(left_inds) && isempty(right_inds) && length(all_inds) == 2
        (all_inds[1:1], all_inds[2:2])
    elseif isempty(left_inds)
        (setdiff(all_inds, right_inds), right_inds)
    elseif isempty(right_inds)
        (left_inds, setdiff(all_inds, left_inds))
    else
        (left_inds, right_inds)
    end

    if !all(!isempty, (left_inds, right_inds))
        throw(ArgumentError("no right-indices left in factorization"))
    end

    if !all(∈(all_inds), left_inds ∪ right_inds)
        throw(ArgumentError("indices must be in $(all_inds)"))
    end

    return left_inds, right_inds
end
