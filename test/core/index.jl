using Test
using Muscle
using Muscle: Invariant, variance, label, is_equal_label

@test adjoint(Covariant) == Contravariant
@test adjoint(Contravariant) == Covariant
@test adjoint(Invariant) == Invariant

i = Index(:i, Covariant)
@test label(i) == Label(:i)
@test variance(i) == Covariant
@test adjoint(i) == Index(:i, Contravariant)

j = Index(:j, Contravariant)
@test label(j) == Label(:j)
@test variance(j) == Contravariant
@test adjoint(j) == Index(:j, Covariant)

k = Index(:k, Invariant)
@test label(k) == Label(:k)
@test variance(k) == Invariant
@test adjoint(k) == Index(:k, Invariant)

is_equal_label(Index(:i, Covariant), Index(:i, Contravariant)) == true
is_equal_label(Index(:i, Covariant), Index(:j, Covariant)) == false
is_equal_label(Index(:i, Covariant), Index(:j, Contravariant)) == false
