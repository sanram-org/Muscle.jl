using Base: @nospecializeinfer

@nospecializeinfer function check_binary_einsum(@nospecialize(a), @nospecialize(b), @nospecialize(contracting_dims), @nospecialize(batching_dims))
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
    
    @assert allequal((ai, bi) -> size(a, ai) == size(b, bi), zip(contracting_dims[1], contracting_dims[2]))
    @assert allequal((ai, bi) -> size(a, ai) == size(b, bi), zip(batching_dims[1], batching_dims[2]))
end

function binary_einsum(@nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); @nospecialize(contracting_dims), batching_dims=((),()))
    check_binary_einsum(a, b, contracting_dims, batching_dims)
    _platform = promote_platform(platform(a), platform(b))
    backend = getbackend(binary_einsum, _platform)
    return binary_einsum(backend, a, b; contracting_dims, batching_dims)
end

@nospecializeinfer function binary_einsum(::BackendBase, @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); contracting_dims, batching_dims)
    @assert isempty(batching_dims[1]) "Batch `binary_einsum` not yet supported in BackendBase"

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

function binary_einsum!(@nospecialize(c::AbstractArray), @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); @nospecialize(contracting_dims), batching_dims=((),()))
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

@nospecializeinfer function binary_einsum!(@nospecialize(B::Backend), @nospecialize(c::AbstractArray), @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); @nospecialize(contracting_dims), @nospecialize(batching_dims))
    @debug "Fallback to generic `binary_einsum!` implementation for backend $B with intermediate copying"
    _c = binary_einsum(B, a, b; contracting_dims, batching_dims)
    copyto!(parent(c), parent(_c))
    return c
end

@nospecializeinfer function binary_einsum!(::BackendBase, @nospecialize(c::AbstractArray), @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); contracting_dims, batching_dims)
    @assert isempty(batching_dims[1]) "Batch `binary_einsum` not yet supported in BackendBase"

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

@nospecializeinfer function check_factorization(@nospecialize(a::AbstractArray), @nospecialize(dims))
    @assert dims isa Base.AbstractVecOrTuple{Base.AbstractVecOrTuple}
    @assert length(dims) == 2
    @assert length(dims[1]) == length(dims[2])

    @assert all(∈(1:ndims(a)), dims[1])
    @assert all(∈(1:ndims(a)), dims[2])
end

@nospecializeinfer function tensor_qr(@nospecialize(a::AbstractArray); @nospecialize(dims), kwargs...)
    check_factorization(a, dims)
    backend = getbackend(tensor_qr, platform(a))
    return tensor_qr(backend, a; dims, kwargs...)
end

@nospecializeinfer function tensor_qr(::BackendBase, @nospecialize(a::AbstractArray); @nospecialize(dims), @nospecialize(kwargs...))
    inds_q, inds_r = dims

    # permute array
    left_sizes = Int[size(a, i) for i in inds_q]
    right_sizes = Int[size(a, i) for i in inds_r]
    a_mat = permutedims(a, [inds_q; inds_r])
    a_mat = reshape(parent(a_mat), prod(left_sizes), prod(right_sizes))

    # compute QR
    F = LinearAlgebra.qr(a_mat; kwargs...)
    q, r = Matrix(F.Q), Matrix(F.R)

    # tensorify results
    q = reshape(Q, left_sizes..., size(Q, 2))
    r = reshape(R, size(R, 1), right_sizes...)

    return q, r
end

@nospecializeinfer function tensor_svd(@nospecialize(a::AbstractArray); @nospecialize(dims), kwargs...)
    check_factorization(a, dims)
    backend = getbackend(tensor_svd, platform(a))
    return tensor_svd(backend, a; dims, kwargs...)
end

@nospecializeinfer function tensor_svd(::BackendBase, @nospecialize(a::AbstractArray); @nospecialize(dims), @nospecialize(kwargs...))
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

@nospecializeinfer function tensor_eigen(@nospecialize(a::AbstractArray); @nospecialize(dims), kwargs...)
    check_factorization(a, dims)
    backend = getbackend(tensor_eigen, platform(a))
    return tensor_eigen(backend, a; dims, kwargs...)
end

@nospecializeinfer function tensor_eigen(::BackendBase, @nospecialize(a::AbstractArray); @nospecialize(dims), @nospecialize(kwargs...))
    inds_l, inds_r = dims

    # permute array
    left_sizes = Int[size(a, i) for i in inds_l]
    right_sizes = Int[size(a, i) for i in inds_r]
    a_mat = permutedims(a, [inds_l; inds_r])
    a_mat = reshape(parent(a_mat), prod(left_sizes), prod(right_sizes))

    # compute eigen
    F = LinearAlgebra.eigen(Amat; kwargs...)

    # tensorify results
    Λ = F.values
    #U = reshape(F.vectors, size(F.vectors, 2), right_sizes...)
    U = reshape(F.vectors, left_sizes..., size(F.vectors, 2))

    return Λ, U
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
    kwargs...
)
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
    atol=0.0,
    rtol=0.0,
    maxdim=nothing,
)
    @debug "Fallback to generic `simple_update` implementation for backend $B"

    # contract state tensors
    Θ = binary_einsum(B, a, b; contracting_dims=((dim_bond_a,), (dim_bond_b,)))

    # contract state tensor with gate
    if dim_physical_a > dim_bond_a
        dim_physical_a -= 1
    end
    if dim_physical_b > dim_bond_b
        dim_physical_b -= 1
    end
    dim_physical_b += ndims(a) - 1
    Θ = binary_einsum(B, Θ, g; contracting_dims=((dim_physical_a, dim_physical_b), (dims_g_a[1], dims_g_b[2])))

    # factorize
    dims_outer_a = sizehint!(Int[], ndims(a) - 2)
    dims_outer_b = sizehint!(Int[], ndims(b) - 2)

    for i in 1:ndims(a)
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
        j = i + ndims(a) - 2
        if i > dim_physical_b
            j -= 1
        end
        if i > dim_bond_a
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
    u, s, vt = tensor_svd(B, Θ; dims, atol, rtol)

    # ad-hoc truncation
    if !isnothing(maxdim)
        u = selectdim(u, ndims(u), 1:min(maxdim, length(s)))
        s = selectdim(s, 1, 1:min(maxdim, length(s)))
        vt = selectdim(v, 1, 1:min(maxdim, length(s)))
    end

    normalize && LinearAlgebra.normalize!(S)

    if absorb isa DontAbsorb
        return u, s, vt
    elseif absorb isa AbsorbU
        u = u .* reshape(s, Int[ones(Int, ndims(u)-1); length(s)])
    elseif absorb isa AbsorbV
        vt = vt .* reshape(s, Int[length(s); ones(Int, ndims(vt) - 1)])
    elseif absorb isa AbsorbEqually
        s_sqrt = sqrt.(s)
        u = u .* reshape(s_sqrt, Int[ones(Int, ndims(u)-1); length(s)])
        vt = vt .* reshape(s_sqrt, Int[length(s); ones(Int, ndims(vt) - 1)])
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
