using Test
using Muscle: Tensor, tensor_qr, binary_einsum, isisometry
using Muscle.Testing

a = construct_test_array(ComplexF64, 2, 4, 6, 8)

# throw if dims are out of bounds
@test_throws AssertionError tensor_qr(a; dims=[100])
@test_throws AssertionError tensor_qr(a; dims=[-1])

# throw if no dims left
@test_throws AssertionError tensor_qr(a, dims=[1,2,3,4])
@test_throws AssertionError tensor_qr(a, dims=Int[])

q, r = tensor_qr(a; dims=[[1,2],[3,4]])
q2, r2 = tensor_qr(a; dims=[1,2])
@test q ≈ q2
@test r ≈ r2

@test size(q) == (2, 4, 8)
@test size(r) == (8, 6, 8)

@test isapprox(binary_einsum(q, r; contracting_dims=[[3],[1]]), a)
@test isisometry(Tensor(q), 3)
