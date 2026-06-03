using Test
using Muscle
using Muscle: factordims

@test Invariant' == Invariant
@test Covariant' == Contravariant
@test Contravariant' == Covariant

@test factordims([Covariant, Contravariant]) == ([2], [1])
@test factordims([Contravariant, Covariant]) == ([1], [2])
@test factordims([Covariant, Contravariant, Covariant, Contravariant]) == ([2, 4], [1, 3])
