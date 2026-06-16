using Base: @nospecializeinfer

@nospecializeinfer function check_hadamard(@nospecialize(c::AbstractArray), @nospecialize(a::AbstractArray), @nospecialize(dims))
    # `b` must be broadcastable to `a`
    @assert ndims(c) >= ndims(a) "`a` must be broadcastable to `c`"
    @assert length(dims) == 2
    @assert length(dims[1]) == length(dims[2])
    @assert length(dims[2]) == ndims(a) "all the dimensions of `a` must be mapped to another dimension of `c`"

    @assert all(((dim_c, dim_a),) -> size(c, dim_c) == size(a, dim_a) || size(a, dim_a) == 1, zip(dims[1], dims[2]))
end

hadamard_prepare_dims(dims) = (hadamard_prepare_dims_sub(dims[1]), hadamard_prepare_dims_sub(dims[2]))
hadamard_prepare_dims_sub(dims::Tuple) = collect(dims)
hadamard_prepare_dims_sub(dims::Vector) = dims

function hadamard!(@nospecialize(c::AbstractArray), @nospecialize(a::AbstractArray); dims)
    check_hadamard(c, a, dims)
    _platform = promote_platform(platform(c), platform(a))
    backend = getbackend(hadamard!, _platform)
    return hadamard!(backend, c, a; dims=hadamard_prepare_dims(dims))
end

function hadamard!(::BackendBase, @nospecialize(c::AbstractArray), @nospecialize(a::AbstractArray); dims)
    # check if this is just a tensor-scalar multiplication
    if ndims(a) == 0
        c .*= a
        return c
    end

    perm = sortperm(dims[1])
    dims_c = dims[1][perm]
    dims_a = dims[2][perm]

    if !issorted(dims_a)
        perm = sortperm(dims_a)
        dims_a = dims_a[perm]
        a = permutedims(a, perm)
    end

    # compute the broadcast shape for `a`
    shape_a_bcast = ones(Int, ndims(c))
    for (dim_c, dim_a) in zip(dims_c, dims_a)
        shape_a_bcast[dim_c] = size(a, dim_a)
    end

    # broadcast element-wise multiplication (Hadamard product)
    a_reshaped = reshape(a, Tuple(shape_a_bcast))
    c .*= a_reshaped

    return c
end

@nospecializeinfer function check_binary_einsum(
    @nospecialize(a), @nospecialize(b), @nospecialize(contracting_dims), @nospecialize(batching_dims)
)
    @assert contracting_dims isa Base.AbstractVecOrTuple{Base.AbstractVecOrTuple}
    @assert length(contracting_dims) == 2
    @assert length(contracting_dims[1]) == length(contracting_dims[2])

    @assert batching_dims isa Base.AbstractVecOrTuple{Base.AbstractVecOrTuple}
    @assert length(batching_dims) == 2
    @assert length(batching_dims[1]) == length(batching_dims[2])

    @assert all(∈(1:ndims(a)), contracting_dims[1])
    @assert all(∈(1:ndims(b)), contracting_dims[2])

    @assert all(∈(1:ndims(a)), batching_dims[1])
    @assert all(∈(1:ndims(b)), batching_dims[2])

    @assert allequal(((ai, bi),) -> size(a, ai) == size(b, bi), zip(contracting_dims[1], contracting_dims[2]))
    @assert allequal(((ai, bi),) -> size(a, ai) == size(b, bi), zip(batching_dims[1], batching_dims[2]))
end

function binary_einsum(
    @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); contracting_dims, batching_dims=((), ())
)
    check_binary_einsum(a, b, contracting_dims, batching_dims)
    _platform = promote_platform(platform(a), platform(b))
    backend = getbackend(binary_einsum, _platform)
    return binary_einsum(backend, a, b; contracting_dims, batching_dims)
end

@nospecializeinfer function binary_einsum(
    ::BackendBase, @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); contracting_dims, batching_dims
)
    @assert all(isempty, batching_dims) "Batch `binary_einsum` not yet supported in BackendBase"

    inner_inds_a, inner_inds_b = collect.(contracting_dims)
    outer_inds_a = Int[i for i in 1:ndims(a) if i ∉ inner_inds_a]
    outer_inds_b = Int[i for i in 1:ndims(b) if i ∉ inner_inds_b]

    sizes_left = Int[size(a, i) for i in outer_inds_a]
    sizes_right = Int[size(b, i) for i in outer_inds_b]
    sizes_contract = Int[size(a, i) for i in inner_inds_a]

    a_mat = reshape(parent(permutedims(a, Int[outer_inds_a; inner_inds_a])), prod(sizes_left), prod(sizes_contract))
    b_mat = reshape(parent(permutedims(b, Int[inner_inds_b; outer_inds_b])), prod(sizes_contract), prod(sizes_right))

    c_mat = a_mat * b_mat
    c = reshape(c_mat, sizes_left..., sizes_right...)

    return c
end

function binary_einsum!(
    @nospecialize(c::AbstractArray),
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray);
    @nospecialize(contracting_dims),
    batching_dims=((), ()),
)
    check_binary_einsum(a, b, contracting_dims, batching_dims)
    batch = Int[size(a, i) for i in batching_dims[1]]
    out_a = Int[size(a, i) for i in 1:ndims(a) if i ∉ contracting_dims[1]]
    out_b = Int[size(b, i) for i in 1:ndims(b) if i ∉ contracting_dims[2]]
    @assert size(c) == Tuple(vcat(batch, out_a, out_b))

    _platform = promote_platform(platform(c), platform(a), platform(b))
    backend = getbackend(binary_einsum!, _platform)
    binary_einsum!(backend, c, a, b; contracting_dims, batching_dims)
    return c
end

@nospecializeinfer function binary_einsum!(
    @nospecialize(B::Backend),
    @nospecialize(c::AbstractArray),
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray);
    @nospecialize(contracting_dims),
    @nospecialize(batching_dims)
)
    @debug "Fallback to generic `binary_einsum!` implementation for backend $B with intermediate copying"
    _c = binary_einsum(B, a, b; contracting_dims, batching_dims)
    copyto!(parent(c), parent(_c))
    return c
end

@nospecializeinfer function binary_einsum!(
    ::BackendBase,
    @nospecialize(c::AbstractArray),
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray);
    contracting_dims,
    batching_dims,
)
    @assert all(isempty, batching_dims) "Batch `binary_einsum` not yet supported in BackendBase"

    inner_inds_a, inner_inds_b = collect.(contracting_dims)
    outer_inds_a = Int[i for i in 1:ndims(a) if i ∉ inner_inds_a]
    outer_inds_b = Int[i for i in 1:ndims(b) if i ∉ inner_inds_b]

    sizes_left = Int[size(a, i) for i in outer_inds_a]
    sizes_right = Int[size(b, i) for i in outer_inds_b]
    sizes_contract = Int[size(a, i) for i in inner_inds_a]

    a_mat = reshape(parent(permutedims(a, Int[outer_inds_a; inner_inds_a])), prod(sizes_left), prod(sizes_contract))
    b_mat = reshape(parent(permutedims(b, Int[inner_inds_b; outer_inds_b])), prod(sizes_contract), prod(sizes_right))
    c_mat = reshape(c, prod(sizes_left), prod(sizes_right))

    LinearAlgebra.mul!(c_mat, a_mat, b_mat)
    return c
end

@nospecializeinfer function check_factorization(@nospecialize(a::AbstractArray), dims)
    @assert dims isa Base.AbstractVecOrTuple{Base.AbstractVecOrTuple}
    @assert length(dims) == 2

    @assert all(∈(1:ndims(a)), dims[1])
    @assert all(∈(1:ndims(a)), dims[2])
end

@nospecializeinfer function tensor_qr(@nospecialize(a::AbstractArray); dims, kwargs...)
    dims = factordims(a, dims)
    check_factorization(a, dims)
    backend = getbackend(tensor_qr, platform(a))
    return tensor_qr(backend, a; dims, kwargs...)
end

@nospecializeinfer function tensor_qr(::BackendBase, @nospecialize(a::AbstractArray); dims, kwargs...)
    inds_q, inds_r = factordims(a, dims)

    # permute array
    left_sizes = Int[size(a, i) for i in inds_q]
    right_sizes = Int[size(a, i) for i in inds_r]
    a_mat = permutedims(a, [inds_q; inds_r])
    a_mat = reshape(parent(a_mat), prod(left_sizes), prod(right_sizes))

    # compute QR
    F = LinearAlgebra.qr(a_mat; kwargs...)
    q, r = Matrix(F.Q), Matrix(F.R)

    # tensorify results
    q = reshape(q, left_sizes..., size(q, 2))
    r = reshape(r, size(r, 1), right_sizes...)

    return q, r
end

@nospecializeinfer function tensor_svd(@nospecialize(a::AbstractArray); dims, kwargs...)
    dims = factordims(a, dims)
    check_factorization(a, dims)
    backend = getbackend(tensor_svd, platform(a))
    return tensor_svd(backend, a; dims, kwargs...)
end

@nospecializeinfer function tensor_svd(::BackendBase, @nospecialize(a::AbstractArray); dims, kwargs...)
    inds_u, inds_v = dims

    # permute array
    left_sizes = Int[size(a, i) for i in inds_u]
    right_sizes = Int[size(a, i) for i in inds_v]
    a_mat = permutedims(a, [inds_u; inds_v])
    a_mat = reshape(parent(a_mat), prod(left_sizes), prod(right_sizes))

    # compute SVD
    F = LinearAlgebra.svd(a_mat; kwargs...)

    # tensorify results
    U = reshape(F.U, left_sizes..., size(F.U, 2))
    s = F.S
    Vt = reshape(F.Vt, size(F.Vt, 1), right_sizes...)

    return U, s, Vt
end

@nospecializeinfer function tensor_eigen(@nospecialize(a::AbstractArray); dims, kwargs...)
    dims = factordims(a, dims)
    @assert prod(i -> size(a, i), dims[1]; init=1) == prod(i -> size(a, i), dims[2]; init=1) "Eigendecomposition requires a square matrix"
    check_factorization(a, dims)
    backend = getbackend(tensor_eigen, platform(a))
    return tensor_eigen(backend, a; dims, kwargs...)
end

@nospecializeinfer function tensor_eigen(::BackendBase, @nospecialize(a::AbstractArray); dims, kwargs...)
    inds_l, inds_r = dims

    # permute array
    left_sizes = Int[size(a, i) for i in inds_l]
    right_sizes = Int[size(a, i) for i in inds_r]
    a_mat = permutedims(a, [inds_l; inds_r])
    a_mat = reshape(parent(a_mat), prod(left_sizes), prod(right_sizes))

    # compute eigen
    F = LinearAlgebra.eigen(a_mat; kwargs...)

    # tensorify results
    λ = F.values
    u = reshape(F.vectors, left_sizes..., size(F.vectors, 2))

    return λ, u
end

# absorb behavior trait
# used to keep type-inference happy (`DontAbsorb` returns 3 tensors, while the rest return 2)
abstract type AbsorbBehavior end
struct DontAbsorb <: AbsorbBehavior end
struct AbsorbU <: AbsorbBehavior end
struct AbsorbV <: AbsorbBehavior end
struct AbsorbEqually <: AbsorbBehavior end

@nospecializeinfer function simple_update(
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray),
    @nospecialize(g::AbstractArray);
    dim_physical_a,
    dim_physical_b,
    dim_bond_a,
    dim_bond_b,
    absorb=DontAbsorb,
    kwargs...,
)
    @assert ndims(a) >= 2
    @assert ndims(b) >= 2
    @assert ndims(g) == 4
    @assert dim_physical_a != dim_bond_a
    @assert dim_physical_b != dim_bond_b
    @assert size(a, dim_physical_a) == size(g, 1) == size(g, 3)
    @assert size(a, dim_physical_b) == size(g, 2) == size(g, 4)
    @assert size(a, dim_bond_a) == size(b, dim_bond_b)
    _platform = promote_platform(platform(a), platform(b), platform(g))
    backend = getbackend(simple_update, _platform)
    return simple_update(backend, a, b, g; dim_physical_a, dim_physical_b, dim_bond_a, dim_bond_b, absorb, kwargs...)
end

@nospecializeinfer function simple_update(
    B::Backend,
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray),
    @nospecialize(g::AbstractArray);
    dim_physical_a,
    dim_physical_b,
    dim_bond_a,
    dim_bond_b,
    absorb=DontAbsorb,
    normalize=false,
    maxdim=nothing,
)
    @debug "Fallback to generic `simple_update` implementation for backend $B"

    # contract state tensors
    Θ = binary_einsum(a, b; contracting_dims=((dim_bond_a,), (dim_bond_b,)))

    # contract state tensor with gate
    dim_physical_a_on_Θ = dim_physical_a
    if dim_physical_a > dim_bond_a
        dim_physical_a_on_Θ -= 1
    end
    dim_physical_b_on_Θ = dim_physical_b
    if dim_physical_b > dim_bond_b
        dim_physical_b_on_Θ -= 1
    end
    dim_physical_b_on_Θ += ndims(a) - 1
    dims_g_a_in = 1
    dims_g_b_in = 2
    Θ = binary_einsum(Θ, g; contracting_dims=((dim_physical_a_on_Θ, dim_physical_b_on_Θ), (dims_g_a_in, dims_g_b_in)))

    # factorize
    dims_outer_a = sizehint!(Int[], ndims(a) - 2)
    dims_outer_b = sizehint!(Int[], ndims(b) - 2)

    for i in 1:ndims(a)
        i == dim_physical_a && continue
        i == dim_bond_a && continue
        j = i
        if i > dim_physical_a
            j -= 1
        end
        if i > dim_bond_a
            j -= 1
        end
        push!(dims_outer_a, j)
    end

    for i in 1:ndims(b)
        i == dim_physical_b && continue
        i == dim_bond_b && continue
        j = i + length(dims_outer_a)
        if i > dim_physical_b
            j -= 1
        end
        if i > dim_bond_b
            j -= 1
        end
        push!(dims_outer_b, j)
    end

    # physical output ind of g for a
    push!(dims_outer_a, ndims(Θ) - 1)

    # physical output ind of g for b
    push!(dims_outer_b, ndims(Θ))

    # TODO other factorizations?
    dims = (dims_outer_a, dims_outer_b)
    u, s, vt = tensor_svd(Θ; dims)

    # ad-hoc truncation
    if !isnothing(maxdim)
        u = selectdim(u, ndims(u), 1:min(maxdim, length(s)))
        s = selectdim(s, 1, 1:min(maxdim, length(s)))
        vt = selectdim(vt, 1, 1:min(maxdim, length(s)))
    end

    normalize && LinearAlgebra.normalize!(s)

    # leave dims as originally found
    perm = collect(1:ndims(u))
    filter!(x -> x != dim_physical_a && x != dim_bond_a, perm)
    push!(perm, dim_physical_a)
    push!(perm, dim_bond_a)
    perm = invperm(perm)
    u = permutedims(u, perm)

    perm = collect(1:ndims(vt))
    filter!(x -> x != dim_physical_b && x != dim_bond_b, perm)
    push!(perm, dim_physical_b)
    pushfirst!(perm, dim_bond_b)
    perm = invperm(perm)
    vt = permutedims(vt, perm)

    if absorb isa DontAbsorb
        return u, s, vt
    elseif absorb isa AbsorbU
        shape = ones(Int, ndims(u))
        shape[dim_bond_a] = length(s)
        u = u .* reshape(s, Tuple(shape))
    elseif absorb isa AbsorbV
        shape = ones(Int, ndims(vt))
        shape[dim_bond_b] = length(s)
        vt = vt .* reshape(s, Tuple(shape))
    elseif absorb isa AbsorbEqually
        s_sqrt = sqrt.(s)

        shape = ones(Int, ndims(u))
        shape[dim_bond_a] = length(s)
        u = u .* reshape(s_sqrt, Tuple(shape))

        shape = ones(Int, ndims(vt))
        shape[dim_bond_b] = length(s)
        vt = vt .* reshape(s_sqrt, Tuple(shape))
    end

    return u, vt
end

# error on fallback methods
for (op, args, kwargs) in [
    (:binary_einsum, :(a::AbstractArray, b::AbstractArray), :(contracting_dims, batching_dims)),
    (:tensor_qr, :(A::AbstractArray), :(kwargs...)),
    (:tensor_svd, :(A::AbstractArray), :(kwargs...)),
    (:tensor_eigen, :(A::AbstractArray), :(kwargs...)),
]
    sign_args_nospec = [:(@nospecialize($arg)) for arg in args.args]
    sign_kwargs_nospec = [:($kwarg) for kwarg in kwargs.args]
    @eval @nospecializeinfer function $op(@nospecialize(B::Backend), $(sign_args_nospec...); $(sign_kwargs_nospec...))
        throw(ArgumentError("`" * $(string(op)) * "` not implemented or not loaded for backend $B"))
    end
end
