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
struct ToeplitzPlusHankelLike{Scalar<:Number} <: AbstractMatrix{Scalar}
    U::Matrix{Scalar}
    V::Matrix{Scalar}
    m::Int
    n::Int
    r::Int

    function ToeplitzPlusHankelLike{Scalar}(U, V) where {Scalar<:Number}
        if size(U, 2) != size(V, 2)
            throw(DomainError("size(U,2) != size(V,2)"))
        end
        m = size(U, 1) 
        n = size(V, 1) 
        r = size(U, 2)

        new{Scalar}(
            convert(Matrix{Scalar}, U),
            convert(Matrix{Scalar}, V),
            m,
            n,
            r,
        )
    end
end
function ToeplitzPlusHankelLike{Scalar}(A::Toeplitz; φ::Number = -1) where {Scalar<:Number}
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
    return ToeplitzPlusHankelLike{Scalar}(U, V)
end
function ToeplitzPlusHankelLike{Scalar}(A::Hankel; φ::Number = -1) where {Scalar<:Number}
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
    return ToeplitzPlusHankelLike{Scalar}(U, V)
end
function ToeplitzPlusHankelLike{Scalar}(A::ToeplitzPlusHankel; φ::Number = -1) where {Scalar<:Number}
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
    return ToeplitzPlusHankelLike{Scalar}(U, V)
end
Base.:size(A::ToeplitzPlusHankelLike) = (A.m, A.n)
function Base.:getindex(A::ToeplitzPlusHankelLike, i::Int, j::Int)
# TODO
    return 0
end
