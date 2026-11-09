"""
Interior point method for the p → q norm of sparse non-negative matrices (Pirazzini &
Spielman). Each Newton step reduces to one SDD (Laplacian) system plus two simplex
constraints. Entry point: [`pqnorm`](@ref).
"""
module MatrixNormIPM

using LinearAlgebra
using SparseArrays
using Printf
import Laplacians

export pqnorm, PQNormResult

include("barrier.jl")
include("linsolve.jl")
include("certificate.jl")
include("ipm.jl")
include("neighborhood_dev.jl")

end
