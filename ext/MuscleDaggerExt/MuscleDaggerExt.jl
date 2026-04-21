module MuscleDaggerExt

using Muscle
using Dagger

function __init__()
    Muscle.register_backend!(BackendDagger())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.binary_einsum, BackendDagger())
end

Muscle.Domain(::Type{<:Dagger.DArray}) = Muscle.PlatformDagger()

Dagger.domainchunks(t::Tensor) = Dagger.domainchunks(parent(t))

include("binary_einsum.jl")

end
