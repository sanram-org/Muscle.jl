using Base: @propagate_inbounds
using Base.Broadcast: Broadcasted, ArrayStyle
using LinearAlgebra

"""
    Variance

Enum representing the variance type of an index. It can take the following values:

- `Covariant`, "down" or transforming in the same way as basis vectors
- `Contravariant`, "up" or transforming in the opposite way as basis vectors
- `Invariant` to change of basis
"""
@enum Variance begin
    Covariant
    Contravariant
    Invariant
end

function Base.adjoint(var::Variance)
    if var == Covariant
        return Contravariant
    elseif var == Contravariant
        return Covariant
    else
        return Invariant
    end
end

"""
    Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}

An array-like object with variance information on each dimension (i.e. [`Variance`](@ref)).
"""
struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    vars::Vector{Variance}

    function Tensor(data::A, vars::Vector{Variance}=fill(Invariant, N)) where {T,N,A<:AbstractArray{T,N}}
        if length(vars) != N
            throw(ArgumentError("ndims(data) [$(ndims(data))] must be equal to length(vars) [$(length(vars))]"))
        end
        return new{T,N,A}(data, vars)
    end

    """
        Tensor(data::A, vars) where {T,N,A<:AbstractArray{T,N}}
    """
    function Tensor(data::A, vars) where {T,N,A<:AbstractArray{T,N}}
        return Tensor(data, collect(vars))
    end
end

Tensor(data::AbstractArray{T,0}) where {T} = Tensor(data, ())
Tensor(data::Number) = Tensor(fill(data))

Tensor(x::Tensor) = x
Tensor{T,N,A}(x::Tensor{T,N,A}) where {T,N,A} = x
function Tensor{T,N,A}(x::Tensor) where {T,N,A}
    throw(ArgumentError("Tensor type mismatch: $(typeof(x)) is not a Tensor{T,N,A}"))
end

Tensor(::Tensor, _) = throw(ArgumentError("Can't wrap a `Tensor` with another `Tensor`"))
function Tensor{T,N,A}(::Tensor, _) where {T,N,A}
    throw(ArgumentError("Can't wrap a `Tensor` with another `Tensor`"))
end

Base.parent(t::Tensor) = t.data
variance(x::Tensor) = x.vars
variance(x::Tensor, i::Integer) = x.vars[i]

platform(x::Tensor) = platform(parent(x))

Base.copy(t::Tensor{T,N,<:SubArray{T,N}}) where {T,N} = Tensor(copy(parent(t)), copy(variance(t)))

arraytype(::Type{Tensor{T,N,A}}) where {T,N,A} = A
arraytype(::T) where {T<:Tensor} = arraytype(T)

Base.@nospecializeinfer function index_signature(@nospecialize(t::Tensor))
    chars = Vector{Char}(undef, ndims(t))
    for i in eachindex(chars)
        var = variance(t, i)
        chars[i] = if var == Covariant
            '↑'
        elseif var == Contravariant
            '↓'
        else
            '-'
        end
    end
    return String(chars)
end

Base.print_array(io::IO, tensor::Tensor) = Base.print_array(io, parent(tensor))
function Base.showarg(io::IO, tensor::Tensor, toplevel)
    toplevel || print(io, "::")
    print(io, "Tensor(")
    Base.showarg(io, parent(tensor), false)
    print(io, ")")
    if ndims(tensor) > 0 && any(!=(Invariant), variance(tensor))
        print(io, " with signature $(index_signature(tensor))")
    end
    return nothing
end

"""
    Base.similar(::Tensor{T,N}[, S::Type, dims::Base.Dims{N}; variance])

Return a uninitialize tensor of the same size, eltype and [`variance`](@ref) as `tensor`. If `S` is provided, the eltype of the tensor will be `S`. If `dims` is provided, the size of the tensor will be `dims`.
"""
Base.similar(t::Tensor; variance=variance(t)) = Tensor(similar(parent(t)), variance)
Base.similar(t::Tensor, S::Type; variance=variance(t)) = Tensor(similar(parent(t), S), variance)
function Base.similar(t::Tensor{T,N}, S::Type, dims::Base.Dims{N}; variance=variance(t)) where {T,N}
    return Tensor(similar(parent(t), S, dims), variance)
end
function Base.similar(t::Tensor, ::Type, dims::Base.Dims{N}; kwargs...) where {N}
    throw(DimensionMismatch("`dims` needs to be of length $(ndims(t))"))
end
Base.similar(t::Tensor{T,N}, dims::Base.Dims{N}; variance=variance(t)) where {T,N} = Tensor(similar(parent(t), dims), variance)
function Base.similar(t::Tensor, dims::Base.Dims{N}; kwargs...) where {N}
    throw(DimensionMismatch("`dims` needs to be of length $(ndims(t))"))
end

"""
    Base.zero(tensor::Tensor)

Return a tensor of the same size, eltype and [`variance`](@ref) as `tensor` but filled with zeros.
"""
Base.zero(t::Tensor) = Tensor(zero(parent(t)), variance(t))

Base.:(==)(a::AbstractArray, b::Tensor) = isequal(b, a)
Base.:(==)(a::Tensor, b::AbstractArray) = isequal(a, b)
Base.:(==)(a::Tensor, b::Tensor) = isequal(a, b)
Base.isequal(a::AbstractArray, b::Tensor) = false
Base.isequal(a::Tensor, b::AbstractArray) = false
# TODO compare under same basis/variance?
Base.isequal(a::Tensor, b::Tensor) = variance(a) == variance(b) && parent(a) == parent(b)
Base.isequal(a::Tensor{A,0}, b::Tensor{B,0}) where {A,B} = isequal(only(a), only(b))

Base.isapprox(a::AbstractArray, b::Tensor) = false
Base.isapprox(a::Tensor, b::AbstractArray) = false
Base.isapprox(a::Tensor, b::Tensor; kwargs...) = variance(a) == variance(b) && isapprox(parent(a), parent(b); kwargs...)
Base.isapprox(a::Tensor{T,0}, b::T; kwargs...) where {T} = isapprox(only(a), b; kwargs...)
Base.isapprox(a::T, b::Tensor{T,0}; kwargs...) where {T} = isapprox(b, a; kwargs...)
Base.isapprox(a::Tensor{A,0}, b::Tensor{B,0}; kwargs...) where {A,B} = isapprox(only(a), only(b); kwargs...)

# Iteration interface
Base.IteratorSize(::Type{Tensor{T,N,A}}) where {T,N,A} = Base.IteratorSize(A)
Base.IteratorEltype(::Type{Tensor{T,N,A}}) where {T,N,A} = Base.IteratorEltype(A)

# Indexing interface
Base.IndexStyle(::Type{Tensor{T,N,A}}) where {T,N,A} = IndexStyle(A)
@propagate_inbounds Base.getindex(t::Tensor, i...) = getindex(parent(t), i...)
@propagate_inbounds Base.setindex!(t::Tensor, v, i...) = setindex!(parent(t), v, i...)

Base.firstindex(t::Tensor) = firstindex(parent(t))
Base.lastindex(t::Tensor) = lastindex(parent(t))

# AbstractArray interface
Base.eltype(x::Tensor) = eltype(x.data)

Base.size(t::Tensor) = size(parent(t))
Base.size(t::Tensor, i) = size(parent(t), i)

"""
    Base.length(::Tensor)

Return the length of the underlying array.
"""
Base.length(t::Tensor) = length(parent(t))

Base.axes(t::Tensor) = axes(parent(t))
Base.axes(t::Tensor, d) = axes(parent(t), d)

# StridedArrays interface
Base.strides(t::Tensor) = strides(parent(t))
Base.stride(t::Tensor, i::Integer) = stride(parent(t), i)

Base.unsafe_convert(::Type{Ptr{T}}, t::Tensor{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(t))

Base.elsize(::Type{Tensor{T,N,A}}) where {T,N,A} = Base.elsize(A)

# Broadcasting
Base.BroadcastStyle(::Type{T}) where {T<:Tensor} = ArrayStyle{T}()

function Base.similar(bc::Broadcasted{ArrayStyle{Tensor{T,N,A}}}, ::Type{ElType}) where {T,N,A,ElType}
    # NOTE already checked if dimension mismatch
    # TODO throw on label mismatch?
    tensor = first(arg for arg in bc.args if arg isa Tensor{T,N,A})
    return similar(tensor, ElType)
end

"""
    Base.selectdim(tensor::Tensor, dim::Integer, i)

Return a view of the tensor where the index for dimension `dim` equals `i`.

!!! note

    This method doesn't return a `SubArray`, but a `Tensor` wrapping a `SubArray`.

See also: [`selectdim`](@ref)
"""
Base.selectdim(t::Tensor, d::Integer, i) = Tensor(selectdim(parent(t), d, i), variance(t))

function Base.selectdim(t::Tensor{T,N}, d::Integer, i::Integer) where {T,N}
    data = selectdim(parent(t), d, i)
    vars = Variance[var for (i, var) in enumerate(variance(t)) if i != d]
    return Tensor(data, vars)
end

"""
    Base.permutedims(tensor::Tensor, perm)

Permute the dimensions of `tensor` according to the given permutation `perm`.
"""
function Base.permutedims(t::Tensor, perm)
    vars = Variance[variance(t, i) for i in perm]
    Tensor(permutedims(parent(t), perm), vars)
end

# shortcut for 0-dimensional tensors
Base.permutedims(t::Tensor{T,0}, _) where {T} = t

Base.permutedims!(dest::Tensor, src::Tensor, perm) = permutedims!(parent(dest), parent(src), perm)

"""
    Base.dropdims(tensor::Tensor; dims)

Return a tensor where the dimensions specified by `dims` are removed. `size(tensor, dim) == 1` for each dimension in `dims`.
"""
function Base.dropdims(t::Tensor; dims=tuple(findall(==(1), size(t))...))
    return Tensor(dropdims(parent(t); dims), variance(t)[setdiff(1:ndims(t), dims)])
end

"""
    Base.view(tensor::Tensor, i...)

Return a view of the tensor with the given indices. If a `Pair` is given, the index is replaced by the value of the pair.

!!! note

    This method doesn't return a `SubArray`, but a `Tensor` wrapping a `SubArray`.
"""
Base.view(t::Tensor, i...) = Tensor(view(parent(t), i...), i...)

# NOTE: `conj` is automatically managed because `Tensor` inherits from `AbstractArray`,
# but there is a bug when calling `conj` on `Tensor{T,0}` which makes it return a `Tensor{Tensor{Complex, 0}, 0}`
"""
    Base.conj(::Tensor)

Return the conjugate of the tensor.
"""
Base.conj(x::Tensor{<:Complex,0}) = Tensor(conj(parent(x)), Variance[])

"""
    Base.adjoint(::Tensor)

Return the adjoint of the tensor; i.e. the 

!!! note

    This method doesn't transpose the array.
"""
Base.adjoint(t::Tensor) = Tensor(conj(t), adjoint.(variance(t)))

# NOTE: Maybe use transpose for lazy transposition ?
Base.transpose(t::Tensor{T,1,A}) where {T,A<:AbstractArray{T,1}} = copy(t)
Base.transpose(t::Tensor{T,2,A}) where {T,A<:AbstractArray{T,2}} = Tensor(transpose(parent(t)), reverse(variance(t)))

"""
    extend(tensor::Tensor; [axis=1, size=1, method=:zeros, variance=Invariant])

Expand the tensor by adding a new dimension with the given `size` at the specified `axis`.
Currently the supported methods are `:zeros` and `:repeat`.
"""
function extend(tensor::Tensor, axis=1, size=1, method=:zeros, variance=Invariant)
    array = parent(tensor)
    data = if size == 1
        reshape(array, Base.size(array)[1:(axis - 1)]..., 1, Base.size(array)[axis:end]...)
    elseif method === :zeros
        extend_zeros(array, axis, size)
    elseif method === :repeat
        extend_repeat(array, axis, size)
    else
        # method === :identity ? __extend_identity(array, axis, size) :
        throw(ArgumentError("method \"$method\" is not valid"))
    end

    vars = (Muscle.variance(tensor)[1:(axis - 1)]..., variance, Muscle.variance(tensor)[axis:end]...)
    return Tensor(data, vars)
end

function extend_zeros(array, axis, size)
    new = zeros(eltype(array), Base.size(array)[1:(axis - 1)]..., size, Base.size(array)[axis:end]...)

    view = selectdim(new, axis, 1)
    copy!(view, array)

    return new
end

function extend_repeat(array, axis, size)
    return repeat(
        reshape(array, Base.size(array)[1:(axis - 1)]..., 1, Base.size(array)[axis:end]...);
        outer=(fill(1, axis - 1)..., size, fill(1, ndims(array) - axis + 1)...),
    )
end

# TODO expand on more than 1 axis
"""
    expand(tensor::Tensor, d::Integer, size; method=:zeros)

Pad the tensor along the dimension specified by `ind` to reach new `size`.
Supported methods are `:zeros` and `:rand`.
"""
function expand(tensor::Tensor, d::Integer, _size; method=:zeros)
    @assert size(d) <= _size "New size $_size of dimension $d must be bigger than or equal to $(size(tensor, d))"
    size(d) == _size && return tensor # TODO return copy instead?

    # TODO use `similar` to do just 1 allocation and set via `views`
    pad_size = [i == d ? _size - size(tensor, i) : size(tensor, i) for i in 1:ndims(tensor)]
    pad_data = if method === :zeros
        zeros(eltype(tensor), Tuple(pad_size))
    elseif method === :rand
        rand(eltype(tensor), Tuple(pad_size))
    else
        throw(ArgumentError("method \"$method\" is not valid"))
    end

    # TODO expand in more d?
    new_data = cat(parent(tensor), pad_data; dims=d)
    return Tensor(new_data, variance(tensor))
end

"""
    Base.cat(a::Tensor, b::Tensor; dims)

Concatenate two tensors `a` and `b` along the specified dimensions `dims`.

The variance of the tensors must be equal, otherwise the second tensor will be permuted to match the first one.
"""
function Base.cat(a::Tensor, b::Tensor; dims)
    @assert variance(a) == variance(b) "Variance of tensors must be equal on concatenating dimensions, got $(a_vars) and $(b_vars)"
    # @assert all(i -> size(a, i) == size(b, i), dims) "Sizes of tensors must be equal in all dimensions except for the concatenation dimensions"
    data = cat(parent(a), parent(b); dims)
    return Tensor(data, variance(a))
end

Base.cat(tensor::Tensor) = tensor
Base.cat(tensors::Tensor...; kwargs...) = foldl((a, b) -> cat(a, b; kwargs...), tensors)

# TODO should this be rethought?
LinearAlgebra.opnorm(x::Tensor, p::Real) = opnorm(parent(x), p)

"""
    fuse(tensor, dims)

Fuses `dims` dimensions, leaves them on the right-side internally permuted.
"""
function fuse(tensor::Tensor, dims)
    @assert allunique(dims)
    @assert dims ⊆ 1:ndims(tensor)
    @assert allequal(Base.Fix1(variance, tensor), dims)

    perm = filter(∉(dims), 1:ndims(tensor))
    append!(perm, dims)

    data = perm == 1:ndims(tensor) ? parent(tensor) : permutedims(parent(tensor), perm)
    data = reshape(data, (size(data)[1:(ndims(data) - length(dims))]..., :))

    vars = Variance[variance(tensor, i) for i in 1:ndims(tensor) if i ∉ dims]
    push!(vars, variance(tensor, first(dims)))
    return Tensor(data, vars)
end

function Base._mapreduce_dim(f, op, init, tensor::Tensor, dims)
    Base._mapreduce_dim(f, op, init, parent(tensor), dim.((tensor,), dims))
end

# fix for ambiguity
# function Base._mapreduce_dim(f, op, init::Base._InitialValue, t::Tensor, c::Colon)
#     Base._mapreduce_dim(f, op, init, parent(t), c)
# end

Base._sum(x::Tensor, dims; kwargs...) = Tensor(Base._sum(parent(x), dim.((x,), dims); kwargs...), variance(x))

# TODO isometry with respect to several output dims
isisometry(a::Tensor, dim::Integer; kwargs...) = isisometry(a, (dim,); kwargs...)

function isisometry(a::Tensor, dims; atol::Real=1e-12)
    @assert 1 <= length(dims) < ndims(a)
    @assert allunique(dims)
    @assert all(∈(1:ndims(a)), dims)
    in_dims = filter!(∉(dims), collect(1:ndims(a)))
    b = einsum(a', a; dims = (in_dims, in_dims))
    n = prod(i -> size(a, i), dims)
    b_mat = reshape(parent(b), n, n)
    return isapprox(b_mat, LinearAlgebra.I(n); atol)
end

# ops
function check_compatible_variance(a, b)
    return a == Covariant && b == Contravariant ||
    a == Contravariant && b == Covariant ||
    a == Invariant || b == Invariant
end

factordims(a) = factordims(a, nothing)
factordims(a::Tensor, dims) = factordims(variance(a), dims)

function factordims(vars, ::Nothing)
    left, right = Int[], Int[]
    for (i, var) in enumerate(vars)
        if var == Contravariant
            push!(left, i)
        elseif var == Covariant
            push!(right, i)
        else
            throw(ArgumentError("Cannot infere left and right indices from variance"))
        end
    end
    return left, right
end

function factordims(vars, dims::Base.AbstractVecOrTuple{Int})
    right = Int[i for i in 1:length(vars) if i ∉ dims]
    return vars, right
end

function factordims(vars, dims::Base.AbstractVecOrTuple{Base.AbstractVecOrTuple})
    n = length(vars)
    @assert all(∈(1:n), dims[1])
    @assert all(∈(1:n), dims[2])
    for i in 1:n
        @assert i ∈ dims[1] || i ∈ dims[2]
    end
    return dims
end

"""
    einsum(a::Tensor, b::Tensor; dims::NTuple{2,Tuple})

Perform a binary tensor contraction operation.

# Keyword arguments

    - `dims`: indices to contract over. Defaults to the set intersection of the indices of `a` and `b`.
"""
function einsum(a::Tensor, b::Tensor; dims)
    # check for compatible variance
    # TODO variate if required
    for (ai, bi) in zip(dims[1], dims[2])
        var_a_i = variance(a, ai)
        var_b_i = variance(b, bi)
        if !check_compatible_variance(var_a_i, var_b_i)
            throw(ArgumentError("index $ai ($var_a_i) and $bi ($var_b_i) have incompatible variance"))
        end
    end
    c = binary_einsum(parent(a), parent(b); contracting_dims=dims)
    vars_a = Variance[variance(a, i) for i in dims[1]]
    vars_b = Variance[variance(b, i) for i in dims[2]]
    return Tensor(c, Variance[vars_a; vars_b])
end

"""
    einsum!(c::Tensor, a::Tensor, b::Tensor)

Perform a binary tensor contraction operation between `a` and `b` and store the result in `c`.
"""
function einsum!(c::Tensor, a::Tensor, b::Tensor; dims)
    # TODO variate if required
    for (ai, bi) in zip(dims[1], dims[2])
        var_a_i = variance(a, ai)
        var_b_i = variance(b, bi)
        if !check_compatible_variance(var_a_i, var_b_i)
            throw(ArgumentError("index $ai ($var_a_i) and $bi ($var_b_i) have incompatible variance"))
        end
    end
    binary_einsum!(parent(c), parent(a), parent(b); contracting_dims=dims)
    return c
end

# TODO return as Tensors
"""
    tensor_qr(A::Tensor; dims, kwargs...)

Perform QR factorization on a tensor. `dims` should be of the form `(dims_q, dims_r)` or just `(dims_q...,)`.
If `dims=nothing`, then [`Covariant`](@ref) and [`Contravariant`] dimensions will be used as left- and right-dimensions.
"""
tensor_qr(a::Tensor; dims=factordims(a), kwargs...) = tensor_qr(parent(a); dims, kwargs...)
LinearAlgebra.qr(a::Tensor; kwargs...) = tensor_qr(a; kwargs...)

"""
    Muscle.tensor_svd(A::Tensor; dims, kwargs...)

Perform SVD factorization on a tensor.

# Keyword arguments

  - `inplace`: If `true`, it will use `A` as workspace variable to save space. Defaults to `false`.
  - `kwargs...`: additional keyword arguments to be passed to `LinearAlgebra.svd`.
"""
tensor_svd(a::Tensor; dims=factordims(a), kwargs...) = tensor_svd(parent(a); dims, kwargs...)
LinearAlgebra.svd(a::Tensor; kwargs...) = tensor_svd(a; kwargs...)

# TODO return as Tensors
"""
    Muscle.tensor_eigen(tensor::Tensor; dims, kwargs...)

Perform eigen factorization on a tensor.

# Keyword arguments

  - `inplace`: If `true`, it will use `A` as workspace variable to save space. Defaults to `false`.
  - `kwargs...`: additional keyword arguments to be passed to `LinearAlgebra.eigen`.
"""
tensor_eigen(a::Tensor; dims=factordims(a), kwargs...) = tensor_eigen(parent(a); dims, kwargs...)
LinearAlgebra.eigen(a::Tensor; kwargs...) = tensor_eigen(a; kwargs...)

# TODO return as Tensors
function simple_update(a::Tensor, b::Tensor, g::Tensor; physical_dims, bond_dims, kwargs...)
    # TODO variate if required
    return simple_update(
        parent(a),
        parent(b),
        parent(g);
        dim_physical_a = physical_dims[1],
        dim_physical_b = physical_dims[2],
        dim_bond_a = bond_dims[1],
        dim_bond_b = bond_dims[2],
        kwargs...
    )
end
