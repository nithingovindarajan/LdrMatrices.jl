###############
# DEFINITIONS #
###############
D(n, φ) = [φ^(-(k - 1) / n) for k = 1:n]
Ω(n) = [exp(π * 2im * (k - 1) / n) for k = 1:n]
Ω(n, φ) = φ^(1 / n) * Ω(n)
Ξ(n) = [2 * cos((k * π) / (n + 1)) for k = 1:n]
Κ(n) = [2 * cos((k * π) / (n)) for k = 0:n-1]

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
Base.:getindex(A::Toeplitz, i::Int, j::Int) = A.coeffs[i-j+A.n]
function Base.:+(A::Toeplitz, B::Toeplitz)
    if size(A) != size(B)
        DimensionMismatch()
    else
        coeffs_new = A.coeffs + B.coeffs
        return Toeplitz{eltype(coeffs_new)}(coeffs_new, size(A, 1), size(A, 2))
    end
end


##########################
# Toeplitz-like matrices #
##########################

function ldr_generators_toeplitz_I(A::Toeplitz; φ = -1)
    # associated with lrd eq: Z * A -  A * Z_φ = UV'
    U = [
        1 A.coeffs[A.m]
        zeros(A.m - 1) A.coeffs[1:A.m-1]-φ*A.coeffs[A.n+1:end]
    ]
    V = [
        conj(A.coeffs[end:-1:A.m+1] - A.coeffs[A.n-1:-1:1]) zeros(A.n - 1)
        conj(-φ * A.coeffs[A.n]) 1
    ]
    return U, V
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


########################
# Hankel-like matrices #
########################

function ldr_generators_hankel_I(A::Hankel; φ = -1)
    # associated with lrd eq: Z' * A -  A * Z_φ = UV'
    U = [
        A.coeffs[A.n+1:end]-φ*A.coeffs[1:A.m-1] zeros(A.m - 1)
        -φ*A.coeffs[A.m] 1
    ]
    V = [
        zeros(A.n - 1) conj(A.coeffs[1:A.n-1] - A.coeffs[A.m+1:end])
        1 conj(A.coeffs[A.n])
    ]
    return U, V
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


#################################
# Toeplitz-Hankel-like matrices #
#################################

function ldr_generators_toeplitz_II(A::Toeplitz)
    # associated with lrd eq: Y00 * A -  A * Y11 = UV'
    # U
    u1 = [
        1
        zeros(A.m - 1)
    ]
    u2 = [
        zeros(A.m - 1)
        1
    ]
    u3 = [
        -A.coeffs[1]
        A.coeffs[1:A.m-1] - A.coeffs[2:A.m]
    ]
    u4 = [
        A.coeffs[end-A.m+2:end] - A.coeffs[end-A.m+1:end-1]
        -A.coeffs[end]
    ]
    U = [u1 u2 u3 u4]
    # V    
    v1 = [
        -conj(A.coeffs[A.n-1:-1:1])
        0
    ]
    v2 = [
        0
        -conj(A.coeffs[end:-1:A.m+1])
    ]
    v3 = [
        zeros(A.n - 1)
        1
    ]
    v4 = [
        1
        zeros(A.n - 1)
    ]
    V = [v1 v2 v3 v4]
    return U, V
end


function ldr_generators_hankel_II(A::Hankel)
    # U
    u1 = [
        1
        zeros(A.m - 1)
    ]
    u2 = [
        zeros(A.m - 1)
        1
    ]
    u3 = [
        A.coeffs[end-A.m+2:end] - A.coeffs[end-A.m+1:end-1]
        -A.coeffs[end]
    ]
    u4 = [
        -A.coeffs[1]
        A.coeffs[1:A.m-1] - A.coeffs[2:A.m]
    ]
    U = [u1 u2 u3 u4]
    # V      
    v1 = [
        0
        -conj(A.coeffs[1:A.n-1])
    ]
    v2 = [
        -conj(A.coeffs[end-A.n+2:end])
        0
    ]
    v3 = [
        zeros(A.n - 1)
        1
    ]
    v4 = [
        1
        zeros(A.n - 1)
    ]
    V = [v1 v2 v3 v4]
    return U, V
end

function ldr_generators_toeplitzplushankel(A::ToeplitzPlusHankel)
    # U
    u1 = [
        1
        zeros(A.m - 1)
    ]
    u2 = [
        zeros(A.m - 1)
        1
    ]
    u3 = [
        A.coeffs_hankel[end-A.m+2:end] - A.coeffs_hankel[end-A.m+1:end-1]
        -A.coeffs_hankel[end]
    ]
    u3 += [
        -A.coeffs_toeplitz[1]
        A.coeffs_toeplitz[1:A.m-1] - A.coeffs_toeplitz[2:A.m]
    ]
    u4 = [
        -A.coeffs_hankel[1]
        A.coeffs_hankel[1:A.m-1] - A.coeffs_hankel[2:A.m]
    ]
    u4 += [
        A.coeffs_toeplitz[end-A.m+2:end] - A.coeffs_toeplitz[end-A.m+1:end-1]
        -A.coeffs_toeplitz[end]
    ]
    U = [u1 u2 u3 u4]
    # V      
    v1 = [
        0
        -conj(A.coeffs_hankel[1:A.n-1])
    ]
    v1 += [
        -conj(A.coeffs_toeplitz[A.n-1:-1:1])
        0
    ]
    v2 = [
        -conj(A.coeffs_hankel[end-A.n+2:end])
        0
    ]
    v2 += [
        0
        -conj(A.coeffs_toeplitz[end:-1:A.m+1])
    ]
    v3 = [
        zeros(A.n - 1)
        1
    ]
    v4 = [
        1
        zeros(A.n - 1)
    ]
    V = [v1 v2 v3 v4]

    return U, V
end
