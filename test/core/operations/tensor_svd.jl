using Test
using Muscle: Tensor, tensor_svd, binary_einsum, isisometry
using Muscle.Testing

a = construct_test_array(ComplexF64, 2, 4, 6, 8)

# throw if dims are out of bounds
@test_throws AssertionError tensor_svd(a; dims=[100])
@test_throws AssertionError tensor_svd(a; dims=[-1])

u, s, vt = tensor_svd(a; dims=[[1,2], [3,4]])
u2, s2, vt2 = tensor_svd(a; dims=[1,2])
@test u ≈ u2
@test s ≈ s2
@test vt ≈ vt2

@test size(u) == (2, 4, 8)
@test size(s) == (8,)
@test size(vt) == (8, 6, 8)

s = reshape(s, 1, 1, 8)
a_re = binary_einsum((u .* s), vt; contracting_dims=[[3],[1]])
@test isapprox(a_re, a)
@test isisometry(Tensor(q), 3)
@test isisometry(Tensor(vt), 1)
