# DEVELOPMENT CODE (`opts.nbhd = true`, off by default): Hypatia-style per-edge neighborhood.
#
# With the edge dual predicted by the Newton step, s_e = -∇F_e(x) - H_e(x) Δ_e, the proximity
# of a trial point x′ is prox_e(x′) = ‖∇F_e(x′) + s_e‖_{H_e(x′)⁻¹}; prox_e(x) = ‖Δ_e‖_{H_e} is
# the edge's share of the Newton decrement. H_e is used in the Schur form of `HessData`:
#   Δ_eᵀ H_e Δ_e = ũᵀ S̃_e ũ + hγ (Δγ + cᵀΔw)²,  S̃_e = [r1+e_z -r1; -r1 r1+e_y],  ũ = Δw/w.

"Diagonal excess (e_z, e_y) contributed by edge e."
@inline function _edge_exc(P::Problem, hd::HessData, e::Int)
    r2 = hd.r2[e]
    return P.α * r2 + (1 - P.α), P.β * r2 + (1 - P.β)
end

"""
max_e of the edge share of the squared decrement, from the scaled reduced solution `u` and the
γ-part `bγ` of the right-hand side (same inputs as `decrement2`).
"""
function max_edge_decrement2(sys::NewtonSystem, u::Vector{Float64}, bγ::AbstractVector{Float64})
    P, hd, m = sys.P, sys.hd, sys.P.m
    mx = 0.0
    @inbounds for e in 1:P.s
        ui, uj = u[P.ei[e]], u[m+P.ej[e]]
        ez, ey = _edge_exc(P, hd, e)
        d = ui - uj
        λe2 = hd.r1[e] * d * d + ez * ui * ui + ey * uj * uj + bγ[e]^2 / hd.hγ[e]
        mx = max(mx, λe2)
    end
    return mx
end

"Per-edge predicted duals s_e = -∇F_e(x) - H_e(x) Δ_e, written into `sd` (s × 3)."
function edge_duals!(sd::Matrix{Float64}, P::Problem, x::State, ew::EdgeWork, hd::HessData,
                     Δ::Vector{Float64})
    m, n = P.m, P.n
    α, β = P.α, P.β
    @inbounds for e in 1:P.s
        i, j = P.ei[e], P.ej[e]
        z, y = x.z[i], x.y[j]
        omr = ew.omr[e]
        gz = -(2α / omr + (1 - α)) / z
        gy = -(2β / omr + (1 - β)) / y
        gγ = 2 * ew.γ[e] / ew.S[e]
        ez, ey = _edge_exc(P, hd, e)
        r1, h, cz, cy = hd.r1[e], hd.hγ[e], hd.cz[e], hd.cy[e]
        Δz, Δy, Δγ = Δ[i], Δ[m+j], Δ[m+n+e]
        uz, uy = Δz / z, Δy / y
        Sz = ((r1 + ez) * uz - r1 * uy) / z            # (D⁻¹ S̃ ũ)_z
        Sy = (-r1 * uz + (r1 + ey) * uy) / y
        τ = cz * Δz + cy * Δy + Δγ
        sd[e, 1] = -gz - (Sz + h * cz * τ)
        sd[e, 2] = -gy - (Sy + h * cy * τ)
        sd[e, 3] = -gγ - h * τ
    end
    return sd
end

"max_e prox_e(x′)² at a trial point x′ (requires `edge_terms!` and `hessian_parts!` at x′)."
function max_prox2(P::Problem, xt::State, ewt::EdgeWork, hdt::HessData, sd::Matrix{Float64})
    α, β = P.α, P.β
    mx = 0.0
    @inbounds for e in 1:P.s
        i, j = P.ei[e], P.ej[e]
        z, y = xt.z[i], xt.y[j]
        omr = ewt.omr[e]
        vz = -(2α / omr + (1 - α)) / z + sd[e, 1]
        vy = -(2β / omr + (1 - β)) / y + sd[e, 2]
        vγ = 2 * ewt.γ[e] / ewt.S[e] + sd[e, 3]
        ez, ey = _edge_exc(P, hdt, e)
        r1, h, cz, cy = hdt.r1[e], hdt.hγ[e], hdt.cz[e], hdt.cy[e]
        a, d = r1 + ez, r1 + ey                     # S̃ = [a -r1; -r1 d]
        wz, wy = z * (vz - cz * vγ), y * (vy - cy * vγ)
        q = (d * wz * wz + 2r1 * wz * wy + a * wy * wy) / (a * d - r1 * r1) + vγ * vγ / h
        mx = max(mx, q)
        isfinite(mx) || return Inf
    end
    return mx
end

const NBHD_SCHEDULE = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)

"""
Search the target instead of the step length: for f in `NBHD_SCHEDULE`, take the full Newton
step toward η + f (η_target - η) and accept the first whose trial point lies in the
neighborhood max_e prox_e < `o.nbhd_bound`; otherwise take a damped centering step.
Returns (t, f, η_f), t = 0 on failure.
"""
function _nbhd_step!(xt::State, ewt::EdgeWork, hdt::HessData, sd::Matrix{Float64},
                     P::Problem, x::State, ew::EdgeWork, hd::HessData,
                     Δc::Vector{Float64}, Δt::Vector{Float64}, η::Float64, ηt::Float64,
                     λcent::Float64, o, prof)
    bound2 = o.nbhd_bound^2
    for f in NBHD_SCHEDULE
        ηf = η + f * (ηt - η)
        Δ = Δc .+ ηf .* Δt
        all(isfinite, Δ) || return 0.0, f, ηf
        t = min(1.0, o.step_fraction * _max_step(P, x, Δ))
        prof[:n_ls_evals] += 1
        _set_trial!(xt, P, x, ew, Δ, t, o.soc)
        if edge_terms!(ewt, P, xt)
            hessian_parts!(hdt, P, xt, ewt)
            edge_duals!(sd, P, x, ew, hd, Δ)
            max_prox2(P, xt, ewt, hdt, sd) < bound2 && return t, f, ηf
        end
        prof[:n_ftb_reject] += 1                    # rejected by the neighborhood
    end
    # fallback: damped centering step toward the current η
    Δ = Δc .+ η .* Δt
    t = min(1 / (1 + λcent), o.step_fraction * _max_step(P, x, Δ))
    for _ in 1:60
        prof[:n_ls_evals] += 1
        _set_trial!(xt, P, x, ew, Δ, t, o.soc)
        if edge_terms!(ewt, P, xt)
            prof[:n_damped] += 1
            return t, 0.0, η
        end
        t /= 2
    end
    return 0.0, 0.0, η
end
