# Reference implementations used by the tests and diagnostics (not part of the package).

using PnormIPM, SparseArrays, LinearAlgebra, Random
using JuMP
using PnormIPM: Problem, State

"""γ = ρ - σ at x, in precision T."""
gamma_of(P::Problem, x::State, ::Type{T}) where {T} =
    [T(x.z[P.ei[e]])^T(P.α) * T(x.y[P.ej[e]])^T(P.β) - T(x.σ[e]) for e in 1:P.s]

"""Barrier (6) at x + d in the (z, y, γ) coordinates, generic in the element type (for
ForwardDiff / BigFloat). `d = [dz; dy; dγ]`. (Simplex constraints are equalities: no slack terms.)"""
function barrier_generic(P::Problem, x::State, d::AbstractVector{T}) where {T}
    m, n, s = P.m, P.n, P.s
    α, β = T(P.α), T(P.β)
    z = x.z .+ d[1:m]
    y = x.y .+ d[m+1:m+n]
    γ = gamma_of(P, x, Float64) .+ d[m+n+1:end]
    F = zero(T)
    for e in 1:s
        i, j = P.ei[e], P.ej[e]
        ρ2 = z[i]^(2α) * y[j]^(2β)
        F -= log(ρ2 - γ[e]^2) + (1 - α) * log(z[i]) + (1 - β) * log(y[j])
    end
    return F
end

"""Dense gradient and Hessian from formulas (15)-(25), in precision T."""
function dense_derivatives(P::Problem, x::State, ::Type{T}) where {T}
    m, n, s = P.m, P.n, P.s
    N = m + n
    α, β = T(P.α), T(P.β)
    z, y = T.(x.z), T.(x.y)
    γ = gamma_of(P, x, T)
    H = zeros(T, N + s, N + s)
    g = zeros(T, N + s)
    for e in 1:s
        i, j = P.ei[e], m + P.ej[e]
        zi, yj, ge = z[i], y[P.ej[e]], γ[e]
        ρ2 = zi^(2α) * yj^(2β)
        S = ρ2 - ge^2
        r = ge^2 / ρ2
        omr = S / ρ2
        g[i] -= (2α / omr + (1 - α)) / zi
        g[j] -= (2β / omr + (1 - β)) / yj
        g[N+e] = 2ge / S
        H[i, i] += (2α / omr + 4α^2 * r / omr^2 + (1 - α)) / zi^2
        H[j, j] += (2β / omr + 4β^2 * r / omr^2 + (1 - β)) / yj^2
        hzy = 4α * β * r / (omr^2 * zi * yj)
        H[i, j] += hzy
        H[j, i] += hzy
        H[N+e, N+e] = 2 * (ρ2 + ge^2) / S^2
        hzg = -(4α / zi) * ρ2 * ge / S^2
        hyg = -(4β / yj) * ρ2 * ge / S^2
        H[i, N+e] = H[N+e, i] = hzg
        H[j, N+e] = H[N+e, j] = hyg
    end
    return g, H
end

"""A random strictly interior point (γ of both signs, slacks bounded away from 0)."""
function random_interior(P::Problem; rng=Random.default_rng())
    z = rand(rng, P.m) .+ 0.1
    z ./= sum(z)
    y = rand(rng, P.n) .+ 0.1
    y ./= sum(y)
    σ = [(1 - (2rand(rng) - 1) * 0.9) * z[P.ei[e]]^P.α * y[P.ej[e]]^P.β for e in 1:P.s]
    return State(z, y, σ)
end

"""‖A‖_{p→q} via JuMP + a conic solver, using 3D power cones (two per edge if p > q)."""
function jump_reference(A::SparseMatrixCSC, p, q; optimizer, silent=true)
    m, n = size(A)
    I, J, V = findnz(A)
    s = length(V)
    α = (q - 1) / q
    β = 1 / p
    model = JuMP.Model(optimizer)
    silent && JuMP.set_silent(model)
    @variable(model, z[1:m] >= 0)
    @variable(model, y[1:n] >= 0)
    @variable(model, γ[1:s])
    @constraint(model, sum(z) <= 1)
    @constraint(model, sum(y) <= 1)
    if p == q
        for e in 1:s
            @constraint(model, [z[I[e]], y[J[e]], γ[e]] in JuMP.MOI.PowerCone(α))
        end
    else
        @variable(model, t[1:s])
        for e in 1:s
            @constraint(model, [z[I[e]], y[J[e]], t[e]] in JuMP.MOI.PowerCone(α / (α + β)))
            @constraint(model, [t[e], 1.0, γ[e]] in JuMP.MOI.PowerCone(α + β))
        end
    end
    @objective(model, Max, sum(V[e] * γ[e] for e in 1:s))
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn "reference solver: $(JuMP.termination_status(model))"
    return JuMP.objective_value(model)
end

"""Dense constrained Newton step: solves [H C; Cᵀ 0][Δ; λ] = [b; 0] in precision T."""
function dense_kkt_solve(P::Problem, H::AbstractMatrix{T}, b::AbstractVector) where {T}
    m, n, s = P.m, P.n, P.s
    C = zeros(T, m + n + s, 2)
    C[1:m, 1] .= 1
    C[m+1:m+n, 2] .= 1
    K = [H C; C' zeros(T, 2, 2)]
    sol = K \ vcat(T.(b), zeros(T, 2))
    return sol[1:m+n+s]
end
