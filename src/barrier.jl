# Barrier (6) of the paper, edge by edge. Variables: z ∈ R^m (rows), y ∈ R^n (columns),
# γ ∈ R^s (one per nonzero, "edge"). Unlike the paper, Σz = 1 and Σy = 1 are equalities (they
# are tight at the optimum), which drops the -log(1-Σz) terms that make the Hessian infinitely
# stiff near the optimum. Exponents α = 1/q*, β = 1/p, κ = 1 - α - β ≥ 0 (κ = 0 iff p = q).
# Edge barrier: F_e = -log(ρ² - γ²) - (1-α) log z_i - (1-β) log y_j,  ρ = z_i^α y_j^β.

"""
Problem data after preprocessing: `A` has no zero rows/columns and max entry 1.
Edges are the nonzeros of `A` in CSC (column-major) order.
"""
struct Problem
    m::Int
    n::Int
    s::Int
    ei::Vector{Int}          # row index of edge e
    ej::Vector{Int}          # column index of edge e
    a::Vector{Float64}       # value of edge e (scaled, ≤ 1)
    A::SparseMatrixCSC{Float64,Int}
    p::Float64
    q::Float64
    ps::Float64              # p* = p/(p-1)
    qs::Float64              # q* = q/(q-1)
    α::Float64
    β::Float64
    κ::Float64
    degz::Vector{Int}        # d_i: nonzeros in row i
    degy::Vector{Int}        # e_j: nonzeros in column j
end

function Problem(A::SparseMatrixCSC{Float64,Int}, p::Float64, q::Float64)
    m, n = size(A)
    ei, ej, a = findnz(A)              # column-major order
    s = length(a)
    ps = p / (p - 1)
    qs = q / (q - 1)
    α = (q - 1) / q
    β = 1 / p
    κ = p == q ? 0.0 : max(1 / q - 1 / p, 0.0)
    degz = zeros(Int, m)
    degy = zeros(Int, n)
    @inbounds for e in 1:s
        degz[ei[e]] += 1
        degy[ej[e]] += 1
    end
    return Problem(m, n, s, ei, ej, a, A, p, q, ps, qs, α, β, κ, degz, degy)
end

"""
IPM iterate. Instead of γ we store the cone slack σ_e = ρ_e - γ_e (ρ_e = z_i^α y_j^β): near the
optimum γ_e → ρ_e and recomputing ρ - γ would lose all relative precision. γ is derived as ρ - σ.
"""
mutable struct State
    z::Vector{Float64}
    y::Vector{Float64}
    σ::Vector{Float64}
end

Base.copy(x::State) = State(copy(x.z), copy(x.y), copy(x.σ))

"""
Closed-form analytic center (cf. Lemma 7) of the barrier on {Σz = 1, Σy = 1}:
at γ = 0 the barrier is -Σ_i (1+α) d_i log z_i - Σ_j (1+β) e_j log y_j, so z_i = d_i/s, y_j = e_j/s.
"""
function analytic_center(P::Problem)
    z = P.degz ./ P.s
    y = P.degy ./ P.s
    σ = [z[P.ei[e]]^P.α * y[P.ej[e]]^P.β for e in 1:P.s]     # γ = 0
    return State(z, y, σ)
end

"""
Per-edge quantities at the current point:
`ρ`, `ρ2 = ρ²`, `γ = ρ - σ`, `S = ρ² - γ² = σ(2ρ - σ)`, `r = γ²/ρ²`, `omr = 1 - r`.
"""
struct EdgeWork
    logz::Vector{Float64}
    logy::Vector{Float64}
    ρ::Vector{Float64}
    ρ2::Vector{Float64}
    γ::Vector{Float64}
    S::Vector{Float64}
    r::Vector{Float64}
    omr::Vector{Float64}
end

EdgeWork(P::Problem) = EdgeWork(zeros(P.m), zeros(P.n), zeros(P.s), zeros(P.s), zeros(P.s),
                                zeros(P.s), zeros(P.s), zeros(P.s))

"""Fill `ew` at `x`; returns `false` iff `x` is not strictly feasible."""
function edge_terms!(ew::EdgeWork, P::Problem, x::State)
    @inbounds for i in 1:P.m
        x.z[i] > 0 || return false
        ew.logz[i] = log(x.z[i])
    end
    @inbounds for j in 1:P.n
        x.y[j] > 0 || return false
        ew.logy[j] = log(x.y[j])
    end
    α, β = P.α, P.β
    @inbounds for e in 1:P.s
        ρ = exp(α * ew.logz[P.ei[e]] + β * ew.logy[P.ej[e]])
        σ = x.σ[e]
        c = 2ρ - σ                       # ρ + γ
        (σ > 0 && c > 0) || return false
        S = σ * c
        (S > 0 && isfinite(S)) || return false
        ew.ρ[e] = ρ
        ew.ρ2[e] = ρ * ρ
        γ = ρ - σ
        ew.γ[e] = γ
        ew.S[e] = S
        t = σ / ρ
        ew.omr[e] = t * (2 - t)
        u = γ / ρ
        ew.r[e] = u * u
    end
    return true
end

"""Barrier value F(x) (formula (6) without the simplex terms); requires `edge_terms!` at `x`."""
function barrier_value(P::Problem, x::State, ew::EdgeWork)
    F = 0.0
    @inbounds for e in 1:P.s
        F -= log(ew.S[e])
    end
    @inbounds for i in 1:P.m
        F -= (1 - P.α) * P.degz[i] * ew.logz[i]
    end
    @inbounds for j in 1:P.n
        F -= (1 - P.β) * P.degy[j] * ew.logy[j]
    end
    return F
end

"""Gradient ∇F(x) (formulas (15)-(17) without the simplex terms) into `g = [g_z; g_y; g_γ]`."""
function gradient!(g::Vector{Float64}, P::Problem, x::State, ew::EdgeWork)
    m, n = P.m, P.n
    α, β = P.α, P.β
    @inbounds begin
        fill!(view(g, 1:m+n), 0.0)
        for e in 1:P.s
            i = P.ei[e]
            j = P.ej[e]
            omr = ew.omr[e]
            g[i] -= (2α / omr + (1 - α)) / x.z[i]
            g[m+j] -= (2β / omr + (1 - β)) / x.y[j]
            g[m+n+e] = 2 * ew.γ[e] / ew.S[e]
        end
    end
    return g
end

"""
Hessian in Schur form (Theorem 10): edge weights `r1` and vertex diagonal excess `exc` of
L = D M D, `hγ` = H_γγ, and `cz`, `cy` = H_zγ/H_γγ, H_yγ/H_γγ (`r2` is used by the dev code).
"""
struct HessData
    r1::Vector{Float64}
    r2::Vector{Float64}
    exc::Vector{Float64}
    hγ::Vector{Float64}
    cz::Vector{Float64}
    cy::Vector{Float64}
end

HessData(P::Problem) = HessData(zeros(P.s), zeros(P.s), zeros(P.m + P.n), zeros(P.s), zeros(P.s), zeros(P.s))

function hessian_parts!(hd::HessData, P::Problem, x::State, ew::EdgeWork)
    m = P.m
    α, β, κ = P.α, P.β, P.κ
    fill!(hd.exc, 0.0)
    @inbounds for e in 1:P.s
        i = P.ei[e]
        j = P.ej[e]
        r = ew.r[e]
        omr = ew.omr[e]
        opr = 1 + r
        r1 = 4α * β * r / (omr * opr)
        r2 = 2 / opr + 4κ * r / (omr * opr)
        hd.r1[e] = r1
        hd.r2[e] = r2
        hd.exc[i] += α * r2 + (1 - α)
        hd.exc[m+j] += β * r2 + (1 - β)
        γe = ew.γ[e]
        S = ew.S[e]
        hd.hγ[e] = 2 * (ew.ρ2[e] + γe * γe) / (S * S)
        hd.cz[e] = -(2α / x.z[i]) * γe / opr
        hd.cy[e] = -(2β / x.y[j]) * γe / opr
    end
    return hd
end
