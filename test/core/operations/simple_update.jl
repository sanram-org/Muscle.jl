using Test
using Muscle: Muscle, simple_update, binary_einsum
using LinearAlgebra: LinearAlgebra, norm

# these two tensors represent a MPS factorization of the |00> + |01> + |10> + |11> state with norm=√2
# out1,bond
Γa = [1.0 0.0; 0.0 1.0]
# out2,bond
Γb = [1.0 0.0; 0.0 1.0]

# in1,in2,out1,out2
op_identity = reshape(Array{Float64}(LinearAlgebra.I(4)), 2, 2, 2, 2)

# in1,in2,out1,out2
op_cx = reshape(
    [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 1.0
        0.0 0.0 1.0 0.0
    ],
    2,
    2,
    2,
    2,
)

@testset "apply identity" begin
    U, s, V = simple_update(
        Γa,
        Γb,
        op_identity;
        dim_physical_a=1,
        dim_physical_b=1,
        dim_bond_a=2,
        dim_bond_b=2,
        absorb=Muscle.DontAbsorb(),
    )

    @test U ≈ Γa
    @test V ≈ Γb
    @test s ≈ [1.0, 1.0]

    ψ = let s = reshape(s, 1, 2)
        binary_einsum(U .* s, V; contracting_dims=[[2],[1]])
    end
    @test binary_einsum(ψ, conj(ψ); contracting_dims=[[1,2],[1,2]]) |> only ≈ 2.0
end

@testset "apply cx" begin
    U, s, V = simple_update(
        Γa,
        Γb,
        op_cx;
        dim_physical_a=1,
        dim_physical_b=1,
        dim_bond_a=2,
        dim_bond_b=2,
        absorb=Muscle.DontAbsorb(),
    )

    @test U ≈ Γa
    @test V ≈ 1 / √2 * [1 -1; 1 1] # [Index((; site=2, cut=1)), Index((; bond=(1, 2)))])
    @test s == [√2, 0.0]

    ψ = let s = reshape(s, 1, 2)
        binary_einsum(U .* s, V; contracting_dims=[[2],[1]])
    end
    @test binary_einsum(ψ, conj(ψ); contracting_dims=[[1,2],[1,2]]) |> only ≈ 2.0
end

@testset "apply identity, normalize" begin
    U, s, V = simple_update(
        Γa,
        Γb,
        op_identity;
        dim_physical_a=1,
        dim_physical_b=1,
        dim_bond_a=2,
        dim_bond_b=2,
        absorb=Muscle.DontAbsorb(),
        normalize=true,
    )

    @test U ≈ Γa
    @test V ≈ Γb
    @test s ≈ [1 / √2, 1 / √2]

    ψ = let s = reshape(s, 1, 2)
        binary_einsum(U .* s, V; contracting_dims=[[2],[1]])
    end
    @test binary_einsum(ψ, conj(ψ); contracting_dims=[[1,2],[1,2]]) |> only ≈ 1.0
end

@testset "apply cx, normalize" begin
    U, s, V = simple_update(
        Γa,
        Γb,
        op_cx;
        dim_physical_a=1,
        dim_physical_b=1,
        dim_bond_a=2,
        dim_bond_b=2,
        absorb=Muscle.DontAbsorb(),
        normalize=true,
    )

    @test U ≈ Γa
    @test V ≈ 1 / √2 * [1 -1; 1 1] # [Index((; site=2, cut=1)), Index((; bond=(1, 2)))])
    @test s == [1.0, 0.0]

    ψ = let s = reshape(s, 1, 2)
        binary_einsum(U .* s, V; contracting_dims=[[2],[1]])
    end
    @test binary_einsum(ψ, conj(ψ); contracting_dims=[[1,2],[1,2]]) |> only ≈ 1.0
end

@testset "apply identity, truncate to χ=1" begin
    U, s, V = simple_update(
        Γa,
        Γb,
        op_identity;
        dim_physical_a=1,
        dim_physical_b=1,
        dim_bond_a=2,
        dim_bond_b=2,
        absorb=Muscle.DontAbsorb(),
        maxdim=1,
    )

    @test U ≈ @view Γa[:, 1:1]
    @test V ≈ @view Γb[:, 1:1]
    @test s ≈ [1]
end

@testset "apply cx, truncate to χ=1" begin
    U, s, V = simple_update(
        Γa,
        Γb,
        op_cx;
        dim_physical_a=1,
        dim_physical_b=1,
        dim_bond_a=2,
        dim_bond_b=2,
        absorb=Muscle.DontAbsorb(),
        maxdim=1,
    )

    @test U ≈ @view Γa[:, 1:1]
    @test V ≈ 1 / √2 * [1; 1;;] # [Index((; site=2, cut=1)), Index((; bond=(1, 2)))])
    # @test size(V, Index((; bond=(1, 2)))) == 1
    @test s ≈ [√2]
end

# TODO test better
@testset "apply identity, absorb s to u" begin
    U, V = simple_update(
        Γa,
        Γb,
        op_identity;
        dim_physical_a=1,
        dim_physical_b=1,
        dim_bond_a=2,
        dim_bond_b=2,
        absorb=Muscle.AbsorbU(),
    )

    @test norm(U) ≈ norm(binary_einsum(Γa, Γb; contracting_dims=[[2],[2]]))
end

# TODO test better
@testset "apply identity, absorb s to v" begin
    U, V = simple_update(
        Γa,
        Γb,
        op_identity;
        dim_physical_a=1,
        dim_physical_b=1,
        dim_bond_a=2,
        dim_bond_b=2,
        absorb=Muscle.AbsorbV(),
    )

    @test norm(U) ≈ norm(binary_einsum(Γa, Γb; contracting_dims=[[2],[2]]))
end

# TODO test better
@testset "apply identity, absorb s equally" begin
    U, V = simple_update(
        Γa,
        Γb,
        op_identity;
        dim_physical_a=1,
        dim_physical_b=1,
        dim_bond_a=2,
        dim_bond_b=2,
        absorb=Muscle.AbsorbEqually(),
    )

    @test norm(U) ≈ norm(V)
    @test norm(binary_einsum(Γa, Γb; contracting_dims=[[2],[2]])) ≈ norm(binary_einsum(U, V; contracting_dims=[[2],[2]]))
end
