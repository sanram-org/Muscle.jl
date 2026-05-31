using Test
using Muscle
using Muscle.Testing
using LinearAlgebra

# TODO numeric test with non-random data
# TODO test on NVIDIA GPU

@testset "$(typeof(alg)) - $T - $(Asize)" for
    alg in [LinearAlgebra.QRIteration(), LinearAlgebra.DivideAndConquer()],
    T in [Float64, ComplexF64],
    Asize in [(4,4), (4,5), (5,4)]

    A = Tensor(construct_test_array(T, Asize...), [Index(:i), Index(:j)])

    U, Σ, Vt = Muscle.tensor_svd(A; inds_u=[Index(:i)], ind_s=Index(:x), alg)

    F = LinearAlgebra.svd(parent(A); alg)
    Uref = Tensor(F.U, [Index(:i), Index(:x)])
    Σref = Tensor(F.S, [Index(:x)])
    Vtref = Tensor(F.Vt, [Index(:x), Index(:j)])

    @test isapprox(U, Uref)
    @test isapprox(Σ, Σref)
    @test isapprox(Vt, Vtref)

    Areconstructed = binary_einsum(hadamard(U, Σ), Vt)
    @test isapprox(A, Areconstructed)
end

A = Tensor(rand(ComplexF64, 2, 4, 6, 8), [Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if inds_u is not provided
@test_throws ArgumentError Muscle.tensor_svd(A)

# throw if index is not present
@test_throws ArgumentError Muscle.tensor_svd(A; inds_u=[Index(:z)])
@test_throws ArgumentError Muscle.tensor_svd(A; inds_v=[Index(:z)])

# throw if no inds left
@test_throws ArgumentError Muscle.tensor_svd(A; inds_u=[Index(:i), Index(:j), Index(:k), Index(:l)])
@test_throws ArgumentError Muscle.tensor_svd(A; inds_v=[Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if chosen virtual index already present
@test_throws ArgumentError Muscle.tensor_svd(A; inds_u=[Index(:i)], ind_s=Index(:j))

U, s, Vt = Muscle.tensor_svd(A; inds_u=[Index(:i), Index(:j)], ind_s=Index(:x))

@test inds(U) == [Index(:i), Index(:j), Index(:x)]
@test inds(s) == [Index(:x)]
@test inds(Vt) == [Index(:x), Index(:k), Index(:l)]

@test size(U, Index(:i)) == 2
@test size(U, Index(:j)) == 4
@test size(U, Index(:x)) == 8

@test size(s) == (8,)

@test size(Vt, Index(:k)) == 6
@test size(Vt, Index(:l)) == 8
@test size(Vt, Index(:x)) == 8

@test isapprox(Muscle.binary_einsum(Muscle.hadamard(U, s), Vt), A)
@test isisometry(U, Index(:x))
@test isisometry(Vt, Index(:x))
