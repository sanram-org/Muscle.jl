module Muscle

using Compat

include("Utils.jl")
include("Backend.jl")
include("Operations.jl")
include("BackendConfig.jl")
include("Tensor.jl")
include("Testing.jl")

export variance, Covariant, Contravariant, Invariant
export Tensor, isisometry

# precompilation
using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # `binary_einsum` instances
        for (Ta, Tb) in [(Float64, Float64), (ComplexF64, ComplexF64), (Float64, ComplexF64), (ComplexF64, Float64)]
            a = Tensor(ones(Ta, 2, 2), [:i, :j])
            b = Tensor(ones(Tb, 2, 2), [:j, :k])
            Muscle.binary_einsum(a, b)

            a = Tensor(ones(Ta, 2, 2, 2), [:i, :j, :k])
            b = Tensor(ones(Tb, 2, 2, 2), [:j, :k, :l])
            Muscle.binary_einsum(a, b)
        end
    end
end

end
