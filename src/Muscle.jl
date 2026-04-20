module Muscle

using Compat
using Reexport

include("Utils/Utils.jl")

include("Index.jl")
export Index, variance, Covariant, Contravariant
@compat public Label, label, Variance, Invariant

include("Tensor.jl")
export Tensor, inds, isisometry

include("Platform.jl")
include("Backend.jl")

include("Operations/Operations.jl")
@reexport using .Operations

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
