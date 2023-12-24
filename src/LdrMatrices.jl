module LdrMatrices


###########
# exports #
###########
export Ω, D, Ξ, Κ
export CauchyLike, Toeplitz, Hankel, ToeplitzPlusHankel
export fast_ge_LU, fast_ge_triangularize, fast_ge_inverse, fast_ge_solve


############
# packages #
############
using LinearAlgebra
using FFTW



include("LdrMatrixTypes.jl")
include("algorithms.jl")

end
