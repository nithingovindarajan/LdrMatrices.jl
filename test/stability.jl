##########################
# Stability benchmarking #
##########################

using LinearAlgebra
using BenchmarkTools
import ToeplitzMatrices
import LdrMatrices

# solve: no pivoting, partial pivoting, complete pivoting