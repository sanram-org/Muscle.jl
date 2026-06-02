module Muscle

using Compat

include("Utils.jl")
include("Backend.jl")
include("Operations.jl")
include("BackendConfig.jl")
include("Tensor.jl")
include("Testing.jl")

export variance, Covariant, Contravariant, Invariant
export Tensor, einsum, einsum!, isisometry

# precompilation
using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # `binary_einsum` instances
        for (Ta, Tb) in [(Float64, Float64), (ComplexF64, ComplexF64), (Float64, ComplexF64), (ComplexF64, Float64)]
            # ij,jk->ik
            a = Tensor(ones(Ta, 2, 2))
            b = Tensor(ones(Tb, 2, 2))
            einsum(a, b; dims=((2,),(1,)))

            # ijk,jkl->il
            a = Tensor(ones(Ta, 2, 2, 2))
            b = Tensor(ones(Tb, 2, 2, 2))
            einsum(a, b; dims=((2,3), (1,2)))
        end
    end
end

end
