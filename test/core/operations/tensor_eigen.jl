using Test
using Muscle: Tensor, tensor_eigen, binary_einsum, isisometry
using Muscle.Testing

a = construct_test_array(ComplexF64, 3, 4, 6, 2)

# throw if dims are out of bounds
@test_throws AssertionError tensor_eigen(a; dims=[100])
@test_throws AssertionError tensor_eigen(a; dims=[-1])

# throw if no dims left
@test_throws AssertionError tensor_eigen(a, dims=[1, 2, 3, 4])
@test_throws AssertionError tensor_eigen(a, dims=Int[])

# throw if non-square 
@test_throws AssertionError tensor_eigen(a; dims=[[1, 3], [2, 4]])

λ, u = tensor_eigen(a; dims=[[1, 2], [3, 4]])
λ2, u2 = tensor_eigen(a; dims=[1, 2])
@test λ ≈ λ2
@test u ≈ u2

@test size(λ) == (12,)
@test size(u) == (3, 4, 12)

uinv = reshape(inv(reshape(u, 12, 12)), 12, 6, 2)
λ = reshape(λ, 1, 1, 12)
a_re = binary_einsum((u .* λ), uinv; contracting_dims=[[3], [1]])

@test isapprox(a_re, a)
