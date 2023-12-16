module LdrMatrices


###########
# exports #
###########
export Ω, D, Ξ, Κ
export CauchyLike, Toeplitz, Hankel, ToeplitzPlusHankel
export fast_LU_cauchy, fast_ge_cauchy, fast_solve_schur


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
Ξ(n) = [2* cos((k*π)/(n+1))  for k = 1:n]
Κ(n) = [2* cos((k*π)/(n)) for k = 0:n-1]


########################
# Cauchy-Like matrices #
########################
struct CauchyLike{Scalar<:Number} <: AbstractMatrix{Scalar}
    omega::Vector{Scalar}
    lambda::Vector{Scalar}
    U::Matrix{Scalar}
    V::Matrix{Scalar}
    m::Int
    n::Int
    r::Int

    function CauchyLike{Scalar}(omega, lambda, U, V) where {Scalar<:Number}
        if size(U, 2) != size(V, 2)
            throw(DomainError("size(U,2) != size(V,2)"))
        end
        if length(omega) != length(lambda)
            throw(DomainError("length(omega) != length(lambda)"))
        end
        if size(U, 1) != length(omega)
            throw(DomainError("size(U,1) != length(omega)"))
        end
        if size(V, 1) != length(lambda)
            throw(DomainError("size(V,1) != length(lambda)"))
        end
        m = length(omega)
        n = length(lambda)
        r = size(U, 2)

        new{Scalar}(
            convert(Vector{Scalar}, omega),
            convert(Vector{Scalar}, lambda),
            convert(Matrix{Scalar}, U),
            convert(Matrix{Scalar}, V),
            m,
            n,
            r,
        )
    end

end
function CauchyLike{Scalar}(omega, lambda) where {Scalar<:Number}
    return CauchyLike{Scalar}(
        omega,
        lambda,
        ones(length(omega), 1),
        ones(length(lambda), 1),
    )
end
Base.:size(A::CauchyLike) = (A.m, A.n)
Base.:getindex(A::CauchyLike, i::Int, j::Int) =
    @views dot(A.V[j, :], A.U[i, :]) / (A.omega[i] - A.lambda[j])
Base.:Matrix(A::CauchyLike{Scalar}) where {Scalar<:Number} =
    ((A.U * A.V') ./ (A.omega .- transpose(A.lambda)))


#############################################################################
# Gohberg, Kailath and Olshevsky (Schur) Algorithm for Cauchy-like matrices #
#############################################################################

function normalize(U, V)
    F = qr(U)
    Unorm = F.Q * Matrix(I, size(U)...)
    Vnorm = V * F.R'
    return Unorm, Vnorm
end


function fast_LU_cauchy(A::CauchyLike; row_pivot = true, column_pivot = true, TOL = 1E-15)

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

        # TODO: Givens rotations

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
    lower = UnitLowerTriangular(LU)
    upper = UpperTriangular(LU)

    return Π_1, Π_2, lower, upper

end

function fast_ge_cauchy(
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

function GKOalgorithm_inverse(A::CauchyLike, row_pivot = true, column_pivot = true)
    #TODO 
end

function fast_solve_schur(A::CauchyLike, b::Vector)

    if size(A, 1) == size(A, 2)
        # triangularize
        Upper, Π_2, btilde = fast_ge_cauchy(A, b)
        # solve triangular system
        x = Upper \ btilde
        # inverse permutation
        x[Π_2] = x
    elseif size(A, 1) > size(A, 2)
        error("overdetermined not yet supported")
    else
        error("underdetermined not yet supported")
    end

    return x
end


#####################
# Toeplitz matrices #
#####################
struct Toeplitz{Scalar<:Number} <: AbstractMatrix{Scalar}
    coeffs::Vector{Scalar}
    m::Int
    n::Int
    function Toeplitz{Scalar}(coeffs, m, n) where {Scalar<:Number}
        if length(coeffs) == m + n - 1
            new{Scalar}(convert(Vector{Scalar}, coeffs), m, n)
        else
            DimensionMismatch()
        end
    end
end
function Toeplitz{Scalar}(coeff_fc::Vector, coeff_fr::Vector) where {Scalar<:Number}
    m = length(coeff_fc)
    n = length(coeff_fr) + 1
    coeffs = [reverse(coeff_fr); coeff_fc]
    return Toeplitz{Scalar}(coeffs, m, n)
end
Base.:size(A::Toeplitz) = (A.m, A.n)
Base.:getindex(A::Toeplitz, i::Int, j::Int) = A.coeffs[i-j+A.m]
function Base.:+(A::Toeplitz, B::Toeplitz)
    if size(A) != size(B)
        DimensionMismatch()
    else
        coeffs_new = A.coeffs + B.coeffs
        return Toeplitz{eltype(coeffs_new)}(coeffs_new, size(A, 1), size(A, 2))
    end
end
function ldr_generators_I(A::Toeplitz)
    U = [1 A.coeffs[A.n]; zeros(A.n - 1) A.coeffs[1:A.n-1]+A.coeffs[A.n+1:end]]
    V = [A.coeffs[end:-1:A.n+1]-A.coeffs[A.n-1:-1:1] zeros(A.n - 1); A.coeffs[A.n] 1]
    return U, V
end
function ldr_generators_II(A::Toeplitz)
    #TODO
end
function cauchyform_complex(A::Toeplitz)
    U, V = ldr_generators_I(A)
    n_sqrt = sqrt(A.n)
    U = ifft(n_sqrt .* U, 1)
    V = ifft(n_sqrt .* conj(D(A.n, -1.0 + 0.0im) .* V), 1)
    return CauchyLike{Complex}(Ω(A.n), Ω(A.n, -1.0 + 0.0im), U, V)
end

function fast_solve_schur(A::Toeplitz, b::Vector)

    if size(A, 1) == size(A, 2)

        # Solve Cauchy system
        Ahat = cauchyform_complex(A)
        bhat = sqrt(A.n) * ifft(b)
        xhat = Ahat \ bhat

        # retrieve solution of original system
        x = Diagonal(D(A.n, -1.0 + 0.0im)) * (1 / sqrt(A.n)) * fft(xhat)

    elseif size(A, 1) > size(A, 2)
        error("overdetermined not yet supported")
    else
        error("underdetermined not yet supported")
    end

    return x
end


###################
# Hankel matrices #
###################
struct Hankel{Scalar<:Number} <: AbstractMatrix{Scalar}
    coeffs::Vector{Scalar}
    m::Int
    n::Int
    function Hankel{Scalar}(coeffs, m, n) where {Scalar<:Number}
        if length(coeffs) == m + n - 1
            new{Scalar}(convert(Vector{Scalar}, coeffs), m, n)
        else
            DimensionMismatch()
        end
    end
end
function Hankel{Scalar}(coeff_fr::Vector, coeff_lc::Vector) where {Scalar<:Number}
    m = length(coeff_fr)
    n = length(coeff_lc) + 1
    coeffs = [coeff_fr; coeff_lc]
    return Hankel{Scalar}(coeffs, m, n)
end
Base.:size(A::Hankel) = (A.m, A.n)
Base.:getindex(A::Hankel, i::Int, j::Int) = A.coeffs[i+j-1]
function Base.:+(A::Hankel, B::Hankel)
    if size(A) != size(B)
        DimensionMismatch()
    else
        coeffs_new = A.coeffs + B.coeffs
        return Hankel{eltype(coeffs_new)}(coeffs_new, size(A, 1), size(A, 2))
    end
end



#################################
# Toeplitz-plus-Hankel matrices #
#################################
struct ToeplitzPlusHankel{Scalar<:Number} <: AbstractMatrix{Scalar}
    coeffs_toeplitz::Vector{Scalar}
    coeffs_hankel::Vector{Scalar}
    m::Int
    n::Int
    function ToeplitzPlusHankel{Scalar}(
        coeffs_toeplitz,
        coeffs_hankel,
        m,
        n,
    ) where {Scalar<:Number}
        if length(coeffs_toeplitz) == length(coeffs_hankel) == m + n - 1
            new{Scalar}(
                convert(Vector{Scalar}, coeffs_toeplitz),
                convert(Vector{Scalar}, coeffs_hankel),
                m,
                n,
            )
        else
            DimensionMismatch()
        end
    end

end
function ToeplitzPlusHankel{Scalar}(
    coeff_toeplitz_fc::Vector,
    coeff_toeplitz_fr::Vector,
    coeff_hankel_fr::Vector,
    coeff_hankel_lc::Vector,
) where {Scalar<:Number}
    m = length(coeff_toeplitz_fc)
    n = length(coeff_toeplitz_fr) + 1
    coeffs_toeplitz = [reverse(coeff_toeplitz_fr); coeff_toeplitz_fc]
    coeffs_hankel = [coeff_hankel_fr; coeff_hankel_lc]
    return ToeplitzPlusHankel{Scalar}(coeffs_toeplitz, coeffs_hankel, m, n)
end
Base.:size(A::ToeplitzPlusHankel) = (A.m, A.n)
Base.:getindex(A::ToeplitzPlusHankel, i::Int, j::Int) =
    A.coeffs_hankel[i+j-1] + A.coeffs_toeplitz[i-j+A.m]
function Base.:+(A::Toeplitz, B::Hankel)
    if size(A) != size(B)
        DimensionMismatch()
    else
        elem_type = typeof(A[1, 1] + B[1, 1])
        return ToeplitzPlusHankel{elem_type}(A.coeffs, B.coeffs, size(A, 1), size(A, 2))
    end
end
function Base.:+(A::Hankel, B::Toeplitz)
    if size(A) != size(B)
        DimensionMismatch()
    else
        elem_type = typeof(A[1, 1] + B[1, 1])
        return ToeplitzPlusHankel{elem_type}(B.coeffs, A.coeffs, size(A, 1), size(A, 2))
    end
end
# TODO more toeplitzplushankel additions


end
