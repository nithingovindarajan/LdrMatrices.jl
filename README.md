# LdrMatrices

`LdrMatrices` is Julia package containing highly efficient solvers for matrices of low displacement rank [1]. Specifically, the package includes routines to solve square systems involving Cauchy, Toeplitz, Hankel, and Toeplitz-plus-Hankel matrices. This is achieved through an implementation of the generalized Schur algorithm by first converting all these structured matrices into Cauchy-like matrices using Fourier or other trigonometric transforms [2]. The package enables both the variant that employs Gaussian elimination
with partial pivoting (GEPP), as well as the more numerically robust complete pivoting (GECP)strategy introduced in [4].

## References
1. Kailath, T., & Sayed, A. H. (1995). Displacement structure: theory and applications. SIAM review, 37(3), 297-386.
2. Heinig, G. (1995). Inversion of generalized Cauchy matrices and other classes of structured matrices. In Linear algebra for signal processing (pp. 63-81). Springer New York.
3. Gohberg, I., Kailath, T., & Olshevsky, V. (1995). Fast Gaussian elimination with partial pivoting for matrices with displacement structure. Mathematics of computation, 64(212), 1557-1576.
4. Gu, M. (1998). Stable and efficient algorithms for structured systems of linear equations. SIAM journal on matrix analysis and applications, 19(2), 279-306.
