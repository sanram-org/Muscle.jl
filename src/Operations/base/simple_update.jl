function simple_update(
    ::Backend,
    A::Tensor,
    ind_physical_a::Index,
    B::Tensor,
    ind_physical_b::Index,
    ind_bond_ab::Index,
    G::Tensor,
    ind_physical_g_a::Index,
    ind_physical_g_b::Index;
    normalize::Bool=false,
    absorb::AbsorbBehavior=DontAbsorb(),
    atol::Float64=0.0,
    rtol::Float64=0.0,
    maxdim=nothing,
)
    Θ = binary_einsum(binary_einsum(A, B; dims=[ind_bond_ab]), G; dims=[ind_physical_a, ind_physical_b])
    Θ = replace(Θ, ind_physical_g_a => ind_physical_a, ind_physical_g_b => ind_physical_b)

    inds_u = setdiff(inds(A), [ind_bond_ab])
    inds_v = setdiff(inds(B), [ind_bond_ab])
    ind_s = ind_bond_ab
    U, S, V = tensor_svd(Θ; inds_u, inds_v, ind_s)

    # TODO use low-rank approximations
    # ad-hoc truncation
    if !isnothing(maxdim)
        U = view(U, ind_s => 1:min(maxdim, length(S)))
        S = view(S, ind_s => 1:min(maxdim, length(S)))
        V = view(V, ind_s => 1:min(maxdim, length(S)))
    end

    normalize && LinearAlgebra.normalize!(S)

    if absorb isa DontAbsorb
        return U, S, V
    elseif absorb isa AbsorbU
        U = hadamard!(U, U, S)
    elseif absorb isa AbsorbV
        V = hadamard!(V, V, S)
    elseif absorb isa AbsorbEqually
        S_sqrt = sqrt.(S)
        U = hadamard!(U, U, S_sqrt)
        V = hadamard!(V, V, S_sqrt)
    end

    return U, V
end
