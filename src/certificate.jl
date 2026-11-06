# Certified bounds on ‖A‖_{p→q} at any interior iterate.
#
# Lower: round y to u = (y/Σy)^{1/p} (Lemma 9), or z through Aᵀ; evaluate ‖Au‖_q and polish
# with power-method steps.
# Upper: weak duality for program (5). With w_e in the dual power cone
#     K* = {w ≥ 0 : (w₁/α)^α (w₂/β)^β (w₃/κ)^κ ≥ |w₄|},  w₄,e = -a_e,
# aᵀγ ≤ max_i Σ_{e∋i} w₁,e + max_j Σ_{e∋j} w₂,e + Σ_e w₃,e. We take (w₁, w₂) from the barrier
# gradient, optimise one global scale in closed form, and take w₃ minimal.

_ψ(t::Real, r::Real) = t > 0 ? t^(r - 1) : zero(t)   # duality map of the r-norm (t ≥ 0)

function _normalize_p(u::Vector{Float64}, p::Float64)
    nu = norm(u, p)
    return nu > 0 ? u ./ nu : u
end

"""Duality map for ‖·‖_r applied to a non-negative vector, scaled to avoid overflow."""
function _dualmap(w::Vector{Float64}, r::Float64)
    mx = maximum(w)
    mx > 0 || return w
    return _ψ.(w ./ mx, r)
end

"""One step of the nonlinear power method: u ↦ ψ_{p*}(Aᵀ ψ_q(Au)), normalised."""
function power_step(P::Problem, u::Vector{Float64})
    v = _dualmap(P.A * u, P.q)
    return _normalize_p(_dualmap(P.A' * v, P.ps), P.p)
end

_value(P::Problem, u) = norm(P.A * u, P.q)

"""Returns `(lb, u)` with `‖u‖_p = 1` and `lb = ‖A u‖_q`."""
function lower_bound(P::Problem, x::State; polish::Int=1)
    Y = sum(x.y)
    uy = _normalize_p((x.y ./ Y) .^ (1 / P.p), P.p)
    Z = sum(x.z)
    vz = (x.z ./ Z) .^ (1 / P.qs)
    uz = _normalize_p(_dualmap(P.A' * vz, P.ps), P.p)
    ly, lz = _value(P, uy), _value(P, uz)
    u, lb = ly >= lz ? (uy, ly) : (uz, lz)
    for _ in 1:polish
        u2 = power_step(P, u)
        l2 = _value(P, u2)
        l2 > lb || break
        u, lb = u2, l2
    end
    return lb, u
end

function _logsumexp(v::Vector{Float64})
    mx = maximum(v)
    isfinite(mx) || return mx
    return mx + log(sum(t -> exp(t - mx), v))
end

"""
Upper bound with (w₁, w₂) = -∇F_e - H_e Δ_e (or the plain gradient if `Δ === nothing`). With Δ
the Newton step for some η this is the classical primal-barrier dual estimate, nearly tight
near the central path. Valid for any Δ; `Inf` if some w is not positive.
"""
function upper_bound(P::Problem, x::State, ew::EdgeWork, Δ::Union{Nothing,Vector{Float64}}=nothing)
    α, β, κ = P.α, P.β, P.κ
    m, n, s, N = P.m, P.n, P.s, P.m + P.n
    rowsum = zeros(m)
    colsum = zeros(n)
    logP = Vector{Float64}(undef, s)
    @inbounds for e in 1:s
        i, j = P.ei[e], P.ej[e]
        z, y = x.z[i], x.y[j]
        omr = ew.omr[e]
        w1 = (2α / omr + (1 - α)) / z
        w2 = (2β / omr + (1 - β)) / y
        if Δ !== nothing                      # subtract the local Hessian times Δ_e
            r = ew.r[e]
            S = ew.S[e]
            γ = ew.γ[e]
            ρ2 = ew.ρ2[e]
            hzz = (2α / omr + 4α^2 * r / omr^2 + (1 - α)) / z^2
            hyy = (2β / omr + 4β^2 * r / omr^2 + (1 - β)) / y^2
            hzy = 4α * β * r / (omr^2 * z * y)
            hzγ = -(4α / z) * ρ2 * γ / S^2
            hyγ = -(4β / y) * ρ2 * γ / S^2
            dz, dy, dγ = Δ[i], Δ[m+j], Δ[N+e]
            w1 -= hzz * dz + hzy * dy + hzγ * dγ
            w2 -= hzy * dz + hyy * dy + hyγ * dγ
        end
        (w1 > 0 && w2 > 0) || return Inf
        rowsum[i] += w1
        colsum[j] += w2
        logP[e] = α * log(w1 / α) + β * log(w2 / β)
    end
    logΘ = log(maximum(rowsum) + maximum(colsum))
    loga_P = log.(P.a) .- logP
    if κ == 0
        # need c^{α+β} P_e ≥ a_e for all e; then UB = c Θ
        logc = maximum(loga_P) / (α + β)
        return exp(logc + logΘ)
    else
        # w₃,e(c) = κ (a_e / (c^{α+β} P_e))^{1/κ};  minimise c Θ + c^{-ρ} S₃
        ρ = (α + β) / κ
        logS3 = log(κ) + _logsumexp(loga_P ./ κ)
        logc = (log(ρ) + logS3 - logΘ) / (ρ + 1)
        return exp(logc + logΘ) * (1 + 1 / ρ)
    end
end
