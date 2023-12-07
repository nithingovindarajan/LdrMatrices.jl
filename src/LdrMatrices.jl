module LdrMatrices


###########
# exports #
###########
export CauchyLike, Ω, GKOalgorithm_LU, GKOalgorithm_solve, Toeplitz, cauchyform, D


############
# packages #
############
using LinearAlgebra
using FFTW

###############
# DEFINITIONS #
###############
D(n, φ::Complex) = [φ^(-(k - 1) / n) for k = 1:n]
Ω(n) = [exp(π * 2im * (k - 1) / n) for k = 1:n]
Ω(n, φ) = φ^(1 / n) * Ω(n)


########################
# Cauchy-Like matrices #
########################
struct CauchyLike{Scalar<:Number} <: AbstractMatrix{Scalar}
    omega::Vector{Scalar}
    lambda::Vector{Scalar}
    U::Matrix{Scalar}
    V::Matrix{Scalar}
    n::Int64
    r::Int64

    function CauchyLike{Scalar}(omega, lambda, U, V) where {Scalar<:Number}
        if size(U) != size(V)
            throw(DomainError("size(U) != size(S)"))
        end
        if length(omega) != size(U, 1)
            throw(DomainError("length(omega) != size(R, 1)"))
        end
        if length(lambda) != size(V, 1)
            throw(DomainError("length(lambda) != size(S, 1)"))
        end
        n = length(lambda)
        r = size(U, 2)

        new{Scalar}(
            convert(Vector{Scalar}, omega),
            convert(Vector{Scalar}, lambda),
            convert(Matrix{Scalar}, U),
            convert(Matrix{Scalar}, V),
            n,
            r,
        )
    end

end
Base.:size(A::CauchyLike) = (A.n, A.n)::Tuple{Int64,Int64}
Base.:getindex(A::CauchyLike, i::Int, j::Int) =
    @views dot(A.V[j, :], A.U[i, :]) / (A.omega[i] - A.lambda[j])
Base.:Matrix(A::CauchyLike{TC}) where {TC<:Number} =
    ((A.U * A.V') ./ (A.omega .- transpose(A.lambda)))::Matrix{TC}


#############################################################################
# Gohberg, Kailath and Olshevsky (Schur) Algorithm for Cauchy-like matrices #
#############################################################################

function normalize(U, V)
    F = qr(U)
    Unorm = F.Q * Matrix(I, size(U)...)
    Vnorm = V * F.R'
    return Unorm, Vnorm
end


function GKOalgorithm_LU(A::CauchyLike; row_pivot = true, column_pivot = true, TOL = 1E-15)

    # dimensions
    n = A.n
    r = A.r

    # initialize
    Π_1 = collect(Int64, 1:n)
    Π_2 = collect(Int64, 1:n)
    μ = copy(A.omega)
    ν = copy(A.lambda)
    U, V = normalize(A.U, A.V)
    nrms = norm.(eachrow(V))
    a = Vector{eltype(A)}(undef, n)
    LU = Matrix{eltype(A)}(undef, n, n)

    for k = 1:n-1

        #### Pivot columns ####

        # find column pivot
        if column_pivot
            j = argmax(nrms[k:end])
            j += k - 1
        else
            j = k
        end

        # swap entries
        if j != k
            LU[1:(k-1), k], LU[1:(k-1), j] = LU[1:(k-1), j], LU[1:(k-1), k]
            V[k, :], V[j, :] = V[j, :], V[k, :]
            ν[k], ν[j] = ν[j], ν[k]
            Π_2[k], Π_2[j] = Π_2[j], Π_2[k]
        end

        #### Pivot rows ####

        # retrieve first column
        a[k:end] = (U[k:end, :] * conj(V[k, :])) ./ (μ[k:end] .- ν[k])

        # find row pivot
        if row_pivot
            i = argmax(abs.(a[k:end]))
            i += k - 1
        else
            i = k
        end

        # raise warning in case pivot entry is very small
        if abs(a[k]) < TOL
            @warn "Pivot entry is dangerously small!"
        end

        # swap entries
        if i != k
            a[k], a[i] = a[i], a[k]
            U[k, :], U[i, :] = U[i, :], U[k, :]
            μ[k], μ[i] = μ[i], μ[k]
            Π_1[k], Π_1[i] = Π_1[i], Π_1[k]
            LU[k, 1:(k-1)], LU[i, 1:(k-1)] = LU[i, 1:(k-1)], LU[k, 1:(k-1)]
        end

        #### Schur update ####

        # TODO: first do givens rotations

        # compute multiplier
        alphinv = 1 / a[k]

        # update LU decomposition 
        LU[k, k] = a[k]
        LU[(k+1):end, k] = alphinv * a[(k+1):end]
        LU[k, (k+1):end] = (conj(V[k+1:end, :]) * U[k, :]) ./ (μ[k] .- ν[k+1:end])

        # Compute generators of schur complement
        U[(k+1):end, :] = U[(k+1):end, :] - LU[(k+1):end, k] * transpose(U[k, :])
        V[(k+1):end, :] =
            V[(k+1):end, :] - conj(alphinv) * conj(LU[k, (k+1):end]) * transpose(V[k, :])

        # normalize generators
        if n - k > r
            U[(k+1):end, :], V[(k+1):end, :] = normalize(U[(k+1):end, :], V[(k+1):end, :])
        end

        # compute norms
        nrms[k+1:end] = norm.(eachrow(V[k+1:end, :]))
    end
    # compute last entry of U
    LU[n, n] = dot(V[n, :], U[n, :]) / (μ[n] - ν[n])

    # triangular factors
    Lower = UnitLowerTriangular(LU)
    Upper = UpperTriangular(LU)

    return Π_1, Π_2, Lower, Upper

end

function GKOalgorithm_solve(
    A::CauchyLike,
    b::Vector;
    row_pivot = true,
    column_pivot = true,
    TOL = 1E-15,
)

    # dimensions
    n = A.n
    r = A.r

    # initialize
    Π_1 = collect(Int64, 1:n)
    Π_2 = collect(Int64, 1:n)

    μ = copy(A.omega)
    ν = copy(A.lambda)
    U, V = normalize(A.U, A.V)
    nrms = norm.(eachrow(V))
    a = Vector{eltype(A)}(undef, n)
    btilde = copy(b)
    Upper = UpperTriangular(Matrix{eltype(A)}(undef, n, n)) # must be a better way!

    for k = 1:n-1

        #### Pivot columns ####

        # find column pivot
        if column_pivot
            j = argmax(nrms[k:end])
            j += k - 1
        else
            j = k
        end

        # swap entries
        if j != k
            Upper[1:(k-1), k], Upper[1:(k-1), j] = Upper[1:(k-1), j], Upper[1:(k-1), k]
            V[k, :], V[j, :] = V[j, :], V[k, :]
            ν[k], ν[j] = ν[j], ν[k]
            Π_2[k], Π_2[j] = Π_2[j], Π_2[k]
        end

        #### Pivot rows ####

        # retrieve first column
        a[k:end] = (U[k:end, :] * conj(V[k, :])) ./ (μ[k:end] .- ν[k])

        # find row pivot
        if row_pivot
            i = argmax(abs.(a[k:end]))
            i += k - 1
        else
            i = k
        end

        # raise warning in case pivot entry is very small
        if abs(a[k]) < TOL
            @warn "Pivot entry is dangerously small!"
        end

        # swap entries
        if i != k
            a[k], a[i] = a[i], a[k]
            U[k, :], U[i, :] = U[i, :], U[k, :]
            μ[k], μ[i] = μ[i], μ[k]
            Π_1[k], Π_1[i] = Π_1[i], Π_1[k]
            btilde[k], btilde[i] = btilde[i], btilde[k]
        end

        #### Schur update ####

        # TODO: first do givens rotations

        # compute multiplier
        alphinv = 1 / a[k]
        f = alphinv * a[(k+1):end]

        # update Upper and btilde
        Upper[k, k] = a[k]
        btilde[(k+1):end] = btilde[(k+1):end] - f * btilde[k]
        Upper[k, (k+1):end] = (conj(V[k+1:end, :]) * U[k, :]) ./ (μ[k] .- ν[k+1:end])

        # Compute generators of schur complement
        U[(k+1):end, :] = U[(k+1):end, :] - f * transpose(U[k, :])
        V[(k+1):end, :] =
            V[(k+1):end, :] - conj(alphinv) * conj(Upper[k, (k+1):end]) * transpose(V[k, :])

        # normalize generators
        if n - k > r
            U[(k+1):end, :], V[(k+1):end, :] = normalize(U[(k+1):end, :], V[(k+1):end, :])
        end

        # compute norms
        nrms[k+1:end] = norm.(eachrow(V[k+1:end, :]))
    end
    # compute last entry of U
    Upper[n, n] = dot(V[n, :], U[n, :]) / (μ[n] - ν[n])



    return Upper, Π_2, btilde
end

# #TODO
# function GKOalgorithm_inverse(A::CauchyLike, row_pivot = true, column_pivot = true)
#     
# end


function Base.:\(A::CauchyLike, b::Vector)

    # run GKO algorithm 
    Upper, Π_2, btilde = GKOalgorithm_solve(A, b)
    # solve triangular system
    x = Upper \ btilde
    # inverse permutation
    x[Π_2] = x

    return x
end


#####################
# Toeplitz matrices #
#####################
struct Toeplitz{Scalar<:Number} <: AbstractMatrix{Scalar}

    coeffs::Vector{Scalar}
    n::Int

    function Toeplitz{Scalar}(coeffs, n) where {Scalar<:Number}
        if length(coeffs) == 2 * n - 1
            new{Scalar}(convert(Vector{Scalar}, coeffs), n)
        else
            DimensionMismatch()
        end
    end

end

function Toeplitz{Scalar}(coeff_lt::Vector, coeff_ut::Vector) where {Scalar<:Number}
    n = length(coeff_lt)
    coeffs = [reverse(coeff_ut); coeff_lt]
    return Toeplitz{Scalar}(convert(Vector{Scalar}, coeffs), n)
end

Base.:size(A::Toeplitz) = (A.n, A.n)
Base.:getindex(A::Toeplitz, i::Int, j::Int) = A.coeffs[i-j+A.n]
function cauchyform(A::Toeplitz)
    n = A.n
    coeffs = A.coeffs
    U = sqrt(n) * ifft([1 coeffs[n]; zeros(n - 1) coeffs[1:n-1]+coeffs[n+1:end]], 1)
    V =
        sqrt(n) * ifft(
            conj( D(n,-1.0 + 0.0im) .* [coeffs[end:-1:n+1]-coeffs[n-1:-1:1] zeros(n - 1); coeffs[n] 1]),
            1,
        )

    return CauchyLike{Complex}(Ω(n), Ω(n,-1.0 + 0.0im), U, V)
end

function Base.:\(A::Toeplitz, b::Vector)

    # Solve Cauchy system
    Ahat = cauchyform(A)
    bhat = sqrt(A.n) * ifft(b)
    xhat = Ahat \ bhat

    # retrieve solution of original system
    x = Diagonal(D(A.n,-1.0+0.0im)) * (1 / sqrt(A.n)) * fft(xhat)

    return x
end


# ###################
# # Hankel matrices #
# ###################



# #################################
# # Toeplitz-plus-Hankel matrices #
# #################################



# ########################
# # Vandermonde matrices #
# ########################



# ##################################
# # Chebyshev-Vandermonde matrices #
# ##################################


end
