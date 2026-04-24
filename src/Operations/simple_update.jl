function simple_update end
function simple_update! end

# absorb behavior trait
# used to keep type-inference happy (`DontAbsorb` returns 3 tensors, while the rest return 2)
abstract type AbsorbBehavior end
struct DontAbsorb <: AbsorbBehavior end
struct AbsorbU <: AbsorbBehavior end
struct AbsorbV <: AbsorbBehavior end
struct AbsorbEqually <: AbsorbBehavior end

# TODO automatically move to GPU if G are on CPU?
function simple_update(
    A::Tensor,
    ind_physical_a,
    B::Tensor,
    ind_physical_b,
    ind_bond_ab,
    G::Tensor,
    ind_physical_g_a,
    ind_physical_g_b;
    kwargs...,
)
    _platform = promote_platform(platform(A), platform(B), platform(G))
    backend = getbackend(simple_update, _platform)

    return simple_update(
        backend, A, ind_physical_a, B, ind_physical_b, ind_bond_ab, G, ind_physical_g_a, ind_physical_g_b; kwargs...
    )
end
