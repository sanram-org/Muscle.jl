<picture>
    <source media="(prefers-color-theme: light)" srcset="docs/src/assets/logo-text.svg">
    <source media="(prefers-color-theme: dark)" srcset="docs/src/assets/logo-text-dark.svg">
    <img alt="" width="80%" src="docs/src/assets/logo-text.svg">
</picture>

> :muscle: Muscles power Tensors :muscle:

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mofeing.github.io/Muscle.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mofeing.github.io/Muscle.jl/dev/)
[![Build Status](https://github.com/mofeing/Muscle.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mofeing/Muscle.jl/actions/workflows/CI.yml?query=branch%3Amain)
<!-- [![codecov](https://codecov.io/gh/mofeing/Muscle.jl/branch/main/graph/badge.svg?token=PG757H00RR)](https://codecov.io/gh/mofeing/Muscle.jl) -->

Muscle.jl is a library for manipulation of tensors. It provides a `Tensor` type which wraps together an `AbstractArray` and a list of `Index`.

For example, the following tensor,
$$ T_{ijk} = \begin{cases}
~~~1 \qquad &i=j=k \\
-1 \qquad &\mathrm{otherwise}
\end{cases} $$

can easily be created as,
```julia
T = Tensor(zeros(2,2,2), [Index(:i), Index(:j), Index(:k)]);

for (i,j,k) in eachindex(IndexCartesian(), T)
    T[i,j,k] = i == j == k ? 1 : -1
end
```

A rather more interesting application is tensor contraction. `Index` labels or names are used for automatically matching contracting dimensions.

```julia
julia> a = Tensor(ComplexF64[1 2; 3 4], [Index(:i), Index(:j)])
2×2 Tensor(::Matrix{ComplexF64}) with signature ij:
 1.0+0.0im  2.0+0.0im
 3.0+0.0im  4.0+0.0im

julia> b = Tensor(ones(2,2,2), [Index(:j), Index(:k), Index(:l)])
2×2×2 Tensor(::Array{Float64, 3}) with signature ijk:
[:, :, 1] =
 1.0  1.0
 1.0  1.0

[:, :, 2] =
 1.0  1.0
 1.0  1.0

julia> binary_einsum(a, b)
2×2×2 Tensor(::Array{ComplexF64, 3}) with signature ijk:
[:, :, 1] =
 3.0+0.0im  3.0+0.0im
 7.0+0.0im  7.0+0.0im

[:, :, 2] =
 3.0+0.0im  3.0+0.0im
 7.0+0.0im  7.0+0.0im
```

## Operations

#### `hadamard(!)`

a.k.a. element-wise multiplication.

#### `unary_einsum(!)`

#### `binary_einsum(!)`

Some backends allow for batching indices.

#### `tensor_qr_thin(!)`

#### `tensor_svd_thin(!)`

#### `tensor_trunc_thin(!)`

#### `tensor_eigen_thin(!)`

#### `simple_update(!)`

Although most backends see it as a composite operation (i.e. they will call other operations), cuTensorNet offers it as a primitive.

## Backends

Muscle implements an unconventional backend system in which multiple backends can be used for different `Muscle.Operations` at the same time, as long as they support the same `Platform`.

Currently, Muscle supports the following `Platform`s:

- Host; i.e. CPU
- Reactant
- CUDA
- Dagger

For example, a user may want to use CuTensorNet.jl for `simple_update` and OMEinsum.jl for `unary_einsum` under the CUDA `Platform`. This is easily configured as,

```julia
using Muscle
using Muscle.Operations: setbackend!, simple_update, unary_einsum
Muscle.Operations.setbackend!(simple_update, Muscle.PlatformCUDA(), Muscle.BackendCuTensorNet())
Muscle.Operations.setbackend!(unary_einsum, Muscle.PlatformCUDA(), Muscle.BackendOMEinsum())
```

The currently support table of backends and operations is,

<!-- | **Platform**      | Host | Reactant    | CUDA             | CUDA        | CUDA           | Dagger    | Host / CUDA | -->
|                   | Base | Reactant.jl | CUDA.jl (cuBLAS) | CuTENSOR.jl | CuTensorNet.jl | Dagger.jl | OMEinsum.jl | Strided.jl |
| ----------------- | ---- | ----------- | ---------------- | ----------- | -------------- | --------- | ----------- | ---------- |
| unary_einsum      |      |             |                  |             |                |           | ✅           |            |
| binary_einsum     | ✅    | ✅           | ✅                | ✅           | ✅              | ✅         | ✅           | ✅          |
| hadamard          | ✅    |             |                  |             |                |           |             |            |
| tensor_qr_thin    | ✅    | ⌛           | ⌛                | -           | ✅              |           | -           | -          |
| tensor_svd_thin   | ✅    | ⌛           | ⌛                | -           | ✅              |           | -           | -          |
| tensor_svd_trunc  | ✅    |             |                  | -           |                |           | -           | -          |
| tensor_eigen_thin | ✅    | ⌛           | ⌛                | -           |                |           | -           | -          |
| simple_update     | ✅    | ⌛           | -                | -           | ✅              |           | -           | -          |

| Legend               |
| -------------------- |
| ✅ : implemented      |
| ➖ : doesn't apply    |
| ⌛ : work in progress |
