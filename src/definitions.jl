###############
# DEFINITIONS #
###############

D(n, φ) = [φ^(-(k - 1) / n) for k = 1:n]
Ω(n) = [exp(π * 2im * (k - 1) / n) for k = 1:n]
Ω(n, φ) = φ^(1 / n) * Ω(n)
Ξ(n) = [2 * cos((k * π) / (n + 1)) for k = 1:n]
Κ(n) = [2 * cos((k * π) / (n)) for k = 0:n-1]