# Practical long-step barrier method for program (5).

"""Per-Newton-step log; the fields after `gap` are inner-solver diagnostics."""
struct IterStats
    iter::Int
    η::Float64          # last accepted (centered) η
    η_target::Float64   # η of this Newton step
    λ::Float64          # Newton decrement at the start of the step
    step::Float64
    lb::Float64         # best certified bounds so far
    ub::Float64
    gap::Float64
    inner_calls::Int    # applications of P ≈ L⁻¹
    inner_its::Int      # applications of the approxchol factor (or Cholesky solves)
    cheb_k::Int         # factor applications per inner solve
    kappa::Float64      # estimated cond(P₀L)
    delta_est::Float64  # estimated δ_k
    t_build::Float64
    t_solve::Float64
    time::Float64
end

"""
Result of [`pqnorm`](@ref): certified bounds `value ≤ ‖A‖_{p→q} ≤ upper`, relative `gap`,
maximiser `u` (‖u‖_p = 1), dual vector `v` (‖v‖_{q*} = 1), `status` (`:optimal`, `:max_iter`,
`:time_limit`, `:stalled`, `:closed_form`), `iterations` (Newton steps), per-step `stats`,
and `profile` (time per component and counters).
"""
struct PQNormResult
    value::Float64
    upper::Float64
    gap::Float64
    u::Vector{Float64}
    v::Vector{Float64}
    status::Symbol
    iterations::Int
    time::Float64
    stats::Vector{IterStats}
    profile::Dict{Symbol,Float64}
end

function Base.show(io::IO, r::PQNormResult)
    @printf(io, "PQNormResult(value = %.12g, upper = %.12g, gap = %.2e, status = :%s, iterations = %d, time = %.3fs)",
            r.value, r.upper, r.gap, r.status, r.iterations, r.time)
end

_conj(p::Real) = p == 1 ? Inf : (isinf(p) ? 1.0 : p / (p - 1))

"""Closed-form cases q = 1 and p = ∞ (and A = 0)."""
function _closed_form(A::SparseMatrixCSC{Float64,Int}, p::Float64, q::Float64, t0)
    m, n = size(A)
    if nnz(A) == 0
        u = isinf(p) ? ones(n) : fill(n^(-1 / p), n)
        return PQNormResult(0.0, 0.0, 0.0, u, zeros(m), :closed_form, 0, time() - t0, IterStats[], Dict{Symbol,Float64}())
    end
    if isinf(p)                                  # ‖A‖_{∞→q} = ‖A 1‖_q for A ≥ 0
        u = ones(n)
        w = A * u
        val = norm(w, q)
        v = _normalize_p(_dualmap(w, q), _conj(q))
        return PQNormResult(val, val, 0.0, u, v, :closed_form, 0, time() - t0, IterStats[], Dict{Symbol,Float64}())
    end
    # q == 1: ‖A‖_{p→1} = ‖Aᵀ1‖_{p*}
    c = Vector(A' * ones(m))
    ps = _conj(p)
    val = norm(c, ps)
    if isinf(ps)                                 # p = 1
        u = zeros(n)
        u[argmax(c)] = 1.0
    else
        u = _normalize_p(_dualmap(c, ps), p)
    end
    return PQNormResult(val, val, 0.0, u, ones(m), :closed_form, 0, time() - t0, IterStats[], Dict{Symbol,Float64}())
end

# Bounds for the barrier-growth factor μ (η ← μη), adapted to the steps each target needs.
const MU_MIN = 4.0
const MU_MAX = 1e4

"""
    pqnorm(A, p, q; tol=1e-6, solver=:approxchol, kw...)
    pqnorm(A, p; kw...)          # q = p

Compute ‖A‖_{p→q} for entrywise non-negative `A` and `p ≥ q ≥ 1` with a long-step barrier
method. With status `:optimal`, `value ≤ ‖A‖_{p→q} ≤ upper` and `(upper - value)/value ≤ tol`.

* `solver`: `:approxchol` (Laplacians.jl, default) or `:cholesky` (CHOLMOD), for the SDD
  system of each Newton step.
* `cheb_steps = 0`: Chebyshev iterations on top of approxchol (0 = one application).
* `split = 0`: approxchol edge splitting (more accurate, more expensive).
* `max_iter = 200`, `time_limit = Inf` (seconds).
* `centrality = 0.9`: move the barrier target once the Newton decrement is ≤ this.
* `barrier_growth = 20.0`: initial μ in η ← μη (then adapted within [4, 1e4]).
* `step_fraction = 0.99`: fraction-to-boundary for z, y.
* `polish = 1`: power-method steps on the rounded primal vector.
* `verbose`; `callback(info)` is called at every Newton step.

`opts` (NamedTuple) holds developer options:
* `adaptive_cheb = false`, `cheb_max = 16`, `cheb_trigger = :both` (or `:stall`, `:slow`),
  `cheb_slow = 5`: double `cheb_steps` when the line search stalls or centering is slow.
* [dev] ablations: `soc = true` (second-order correction), `damped`, `cone_ftb`,
  `forced_accept`, `nbhd`, `nbhd_bound = 0.99` (neighborhood_dev.jl), `lanczos_steps = 6`,
  `lanczos_every = 1`.
"""
function pqnorm(A::AbstractMatrix, p::Real, q::Real; tol::Real=1e-6,
                solver::Symbol=:approxchol, cheb_steps::Integer=0, split::Integer=0,
                max_iter::Integer=200, time_limit::Real=Inf, centrality::Real=0.9,
                barrier_growth::Real=20.0, step_fraction::Real=0.99, polish::Integer=1,
                verbose::Bool=false, callback=nothing, opts::NamedTuple=(;))
    t0 = time()
    defaults = (adaptive_cheb=false, cheb_max=16, cheb_trigger=:both, cheb_slow=5,
                soc=true, damped=false, cone_ftb=false, forced_accept=false,  # [dev]
                nbhd=false, nbhd_bound=0.99, lanczos_steps=6, lanczos_every=1, # [dev]
                step_fraction=Float64(step_fraction))
    unknown = setdiff(keys(opts), keys(defaults))
    isempty(unknown) || throw(ArgumentError("unknown opts: $(join(unknown, ", "))"))
    o = merge(defaults, opts)
    tol > 0 || throw(ArgumentError("tol must be > 0, got $tol"))
    0 < step_fraction < 1 || throw(ArgumentError("step_fraction must be in (0, 1), got $step_fraction"))
    0 < centrality < 1 || throw(ArgumentError("centrality must be in (0, 1), got $centrality"))
    barrier_growth > 1 || throw(ArgumentError("barrier_growth must be > 1, got $barrier_growth"))
    max_iter >= 0 || throw(ArgumentError("max_iter must be ≥ 0, got $max_iter"))
    cheb_steps >= 0 || throw(ArgumentError("cheb_steps must be ≥ 0, got $cheb_steps"))
    split >= 0 || throw(ArgumentError("split must be ≥ 0, got $split"))
    polish >= 0 || throw(ArgumentError("polish must be ≥ 0, got $polish"))
    solver in (:cholesky, :approxchol) ||
        throw(ArgumentError("solver must be :approxchol or :cholesky, got :$solver"))
    θ, μ0 = Float64(centrality), Float64(barrier_growth)
    prof = Dict{Symbol,Float64}(k => 0.0 for k in (:t_factor, :t_lanczos, :t_apply, :t_build,
        :t_solve, :t_cert, :t_ls, :n_damped, :n_forced, :n_ftb_reject, :n_ls_evals, :n_escalate,
        :cheb_final))
    p = Float64(p)
    q = Float64(q)
    (q >= 1 && p >= q) || throw(ArgumentError("need p ≥ q ≥ 1 (got p = $p, q = $q)"))
    As = dropzeros(SparseMatrixCSC{Float64,Int}(sparse(A)))
    all(>=(0), nonzeros(As)) || throw(ArgumentError("A must be entrywise non-negative"))
    M, Nc = size(As)
    if nnz(As) == 0 || q == 1 || isinf(p)
        return _closed_form(As, p, q, t0)
    end

    # --- preprocessing: drop zero rows/cols, scale to max entry 1
    rows = findall(>(0), vec(sum(As; dims=2)))
    cols = findall(>(0), vec(sum(As; dims=1)))
    Ar = As[rows, cols]
    Amax = maximum(nonzeros(Ar))
    Ar = Ar ./ Amax
    P = Problem(Ar, p, q)
    m, n, s = P.m, P.n, P.s
    N = m + n

    # --- initialization: analytic center (Lemma 7)
    x = analytic_center(P)
    ew = EdgeWork(P)
    edge_terms!(ew, P, x) || error("analytic center infeasible (should not happen)")
    sys = NewtonSystem(P; solver, cheb_steps, split, lanczos_steps=o.lanczos_steps,
                       lanczos_every=o.lanczos_every)
    # adaptive inner accuracy (opts.adaptive_cheb): cheb_steps 0 → 1 → 2 → 4 … ≤ cheb_max
    escalate_ok(sys) = o.adaptive_cheb && solver === :approxchol && sys.cheb_steps < o.cheb_max
    slow_steps = 0                                # centering steps since the last target / escalation
    function escalate!(sys)
        slow_steps = 0
        sys.cheb_steps = min(o.cheb_max, max(1, 2sys.cheb_steps))
        prof[:n_escalate] += 1
        verbose && println("  (inner accuracy increased: cheb_steps = $(sys.cheb_steps))")
    end
    abar = vcat(zeros(N), P.a)
    g = zeros(N + s)

    η = 0.0
    ηt = 0.0
    μ = Float64(μ0)
    ksteps = 0
    status = :max_iter
    stats = IterStats[]
    xt = copy(x)
    ewt = EdgeWork(P)
    hdt = HessData(P)                             # [dev] Hessian at trial points (nbhd variant)
    sd = zeros(o.nbhd ? s : 0, 3)                 # [dev] per-edge predicted duals (nbhd variant)

    # best certified bounds so far (typed: reassigned in a closure); certify! returns the gap
    best_lb::Float64, best_ub::Float64 = -Inf, Inf
    best_u::Vector{Float64} = zeros(n)
    best_x::State = copy(x)
    function certify!(x, ew)
        tc = time()
        lb, u = lower_bound(P, x; polish)
        if lb > best_lb
            best_lb, best_u, best_x = lb, u, copy(x)
        end
        best_ub = min(best_ub, upper_bound(P, x, ew))
        prof[:t_cert] += time() - tc
        return (best_ub - best_lb) / best_lb
    end

    if verbose
        @printf("pqnorm: m = %d, n = %d, s = %d, p = %g, q = %g, solver = :%s", m, n, s, p, q, solver)
        solver === :approxchol && @printf(", cheb_steps = %d, split = %d", cheb_steps, split)
        println()
        @printf("%5s %10s %10s %9s %8s %15s %15s %9s %6s %6s %8s\n",
                "iter", "η", "η_target", "λ", "step", "lower", "upper", "gap", "inner", "k", "time")
    end

    iters = 0                                     # Newton systems solved
    for it in 1:max_iter
        # ---- certificates at the current point
        gradient!(g, P, x, ew)
        gap = certify!(x, ew)
        if gap <= tol
            status = :optimal
            break
        end

        # ---- adaptive inner accuracy: centering has been slow for this target
        if escalate_ok(sys) && o.cheb_trigger in (:both, :slow) && slow_steps >= o.cheb_slow
            escalate!(sys)
        end

        # ---- Newton system: one factorization, two right-hand sides
        reset_counters!(sys)
        assemble!(sys, x, ew)
        Δc, uc = newton_solve(sys, x, -g)         # centering direction  -H⁻¹∇F (projected)
        Δt, ut = newton_solve(sys, x, abar)       # tangent direction     H⁻¹ā  (projected)
        iters = it
        gγ = @view g[N+1:end]
        λ2(η_) = decrement2(sys, uc .+ η_ .* ut, η_ .* P.a .- gγ)   # squared Newton decrement for Φ_η_

        if it == 1
            # at the analytic center ∇F = 0, so λ_η(x₀) = η ‖Δt‖_H: start where λ = θ
            ηt = θ / sqrt(max(decrement2(sys, ut, P.a), eps()))
        elseif o.nbhd
            # [dev] Hypatia-style: no centering phase; the target moves with every step (below)
        elseif (o.forced_accept && ksteps >= 25) || λ2(ηt) <= θ^2
            λ2(ηt) <= θ^2 || (prof[:n_forced] += 1)   # [dev] forced acceptance
            # centered for the target: accept it and set the next target
            η = ηt
            μ = ksteps <= 2 ? min(2μ, MU_MAX) : (ksteps >= 5 ? max(μ / 2, MU_MIN) : μ)
            ksteps = 0
            slow_steps = 0
            ηt = μ * η
        end

        Δ = Δc .+ ηt .* Δt
        λ = sqrt(λ2(ηt))

        # ---- Newton-corrected dual certificates (free given the solves): s = -∇F - HΔ_η
        tc = time()
        for η_ in (η, ηt)
            η_ > 0 || continue
            best_ub = min(best_ub, upper_bound(P, x, ew, Δc .+ η_ .* Δt))
        end
        prof[:t_cert] += time() - tc
        for k in (:t_factor, :t_lanczos, :t_apply, :t_build, :t_solve)
            prof[k] += getfield(sys, k)
        end
        gap = (best_ub - best_lb) / best_lb
        if gap <= tol
            status = :optimal
            break
        end
        callback === nothing || callback((; it, P, x, ew, g, η, ηt, Δc, Δt, Δ, λ, sys))

        # ---- line search on Φ_{ηt} = F - ηt āᵀx
        tl = time()
        if o.nbhd                                 # [dev] neighborhood variant
            t, f, ηf = _nbhd_step!(xt, ewt, hdt, sd, P, x, ew, sys.hd, Δc, Δt, η, ηt,
                                   sqrt(λ2(η)), o, prof)
            if t > 0
                η = ηf
                μ = f == 1 ? min(2μ, MU_MAX) : (f < 0.5 ? max(μ / 2, MU_MIN) : μ)
                ηt = η > 0 ? μ * η : ηt / 2
            end
        else
            t = _line_search!(xt, ewt, P, x, ew, Δ, ηt, λ, o, prof)
        end
        prof[:t_ls] += time() - tl
        push!(stats, _stat(it, η, ηt, λ, t, best_lb, best_ub, gap, sys, Amax, t0))
        verbose && _print_stat(stats[end])

        if t == 0.0                               # no acceptable step
            if escalate_ok(sys) && o.cheb_trigger in (:both, :stall)
                escalate!(sys)                    # redo this step with a more accurate operator
                continue
            end
            status = :stalled
            break
        end
        x, xt = xt, x
        ew, ewt = ewt, ew
        ksteps += 1
        slow_steps += 1
        if time() - t0 > time_limit
            status = :time_limit
            break
        end
    end

    # final certificate at the last iterate (if the loop ran out)
    if status !== :optimal
        certify!(x, ew) <= tol && (status = :optimal)
    end
    gap = (best_ub - best_lb) / best_lb
    prof[:cheb_final] = sys.cheb_steps

    ufull = zeros(Nc)
    ufull[cols] .= best_u
    vfull = zeros(M)
    vfull[rows] .= (best_x.z ./ sum(best_x.z)) .^ (1 / P.qs)
    res = PQNormResult(best_lb * Amax, best_ub * Amax, gap, ufull, vfull, status, iters,
                       time() - t0, stats, prof)
    verbose && println(res)
    return res
end

pqnorm(A::AbstractMatrix, p::Real; kw...) = pqnorm(A, p, p; kw...)

function _stat(it, η, ηt, λ, t, lb, ub, gap, sys, Amax, t0)
    IterStats(it, η, ηt, λ, t, lb * Amax, ub * Amax, gap, sys.inner_calls, sys.inner_its,
              sys.cheb_k, sys.kappa, sys.delta_est,
              sys.t_build, sys.t_solve, time() - t0)
end

function _print_stat(st::IterStats)
    @printf("%5d %10.3e %10.3e %9.2e %8.2e %15.9e %15.9e %9.2e %6d %6d %8.3f\n",
            st.iter, st.η, st.η_target, st.λ, st.step, st.lb, st.ub, st.gap,
            st.inner_its, st.cheb_k, st.time)
end

"""
Trial point x + tΔ with a second-order correction: z, y move linearly and the cone slack
σ = ρ - γ moves along its linearization Δσ = ρ (α Δz/z + β Δy/y) - Δγ, so the curvature of
ρ = z^α y^β goes into γ. Without it, edges far below their central slack cannot recover
(Maratos effect; the method stalls with λ ≈ √3).
"""
function _set_trial!(xt::State, P::Problem, x::State, ew::EdgeWork, Δ::Vector{Float64},
                     t::Float64, soc::Bool=true)
    m, n = P.m, P.n
    α, β = P.α, P.β
    @inbounds begin
        for i in 1:m
            xt.z[i] = x.z[i] + t * Δ[i]
        end
        for j in 1:n
            xt.y[j] = x.y[j] + t * Δ[m+j]
        end
        for e in 1:P.s
            i, j = P.ei[e], P.ej[e]
            if soc
                dσ = ew.ρ[e] * (α * Δ[i] / x.z[i] + β * Δ[m+j] / x.y[j]) - Δ[m+n+e]
                xt.σ[e] = x.σ[e] + t * dσ
            else                                   # [dev] soc = false: plain x + tΔ, γ moves linearly
                dρ = ew.ρ[e] * expm1(α * log1p(t * Δ[i] / x.z[i]) + β * log1p(t * Δ[m+j] / x.y[j]))
                xt.σ[e] = x.σ[e] + dρ - t * Δ[m+n+e]
            end
        end
    end
    return xt
end

"""Φ_η(xt) - Φ_η(x) for Φ_η = F - η aᵀγ, from log-ratios (Φ is O(η), the change O(λ²))."""
function _phi_change(P::Problem, x::State, ew::EdgeWork, xt::State, ewt::EdgeWork,
                     Δ::Vector{Float64}, t::Float64, η::Float64)
    m, n, N = P.m, P.n, P.m + P.n
    d = 0.0
    @inbounds begin
        for i in 1:m
            d -= (1 - P.α) * P.degz[i] * log1p(t * Δ[i] / x.z[i])
        end
        for j in 1:n
            d -= (1 - P.β) * P.degy[j] * log1p(t * Δ[m+j] / x.y[j])
        end
        for e in 1:P.s
            σ, σt = x.σ[e], xt.σ[e]
            c, ct = 2ew.ρ[e] - σ, 2ewt.ρ[e] - σt
            d -= log1p((σt - σ) / σ) + log1p((ct - c) / c)
            i, j = P.ei[e], P.ej[e]
            dρ = ew.ρ[e] * expm1(P.α * log1p(t * Δ[i] / x.z[i]) + P.β * log1p(t * Δ[m+j] / x.y[j]))
            d -= η * P.a[e] * (dρ - (σt - σ))
        end
    end
    return d
end

"""[dev] `opts.cone_ftb`: each factor of the cone slack σ(2ρ - σ) keeps ≥ `frac` of its value."""
function _keeps_distance(P::Problem, x::State, ew::EdgeWork, xt::State, ewt::EdgeWork; frac=0.01)
    @inbounds for e in 1:P.s
        xt.σ[e] >= frac * x.σ[e] || return false
        (2ewt.ρ[e] - xt.σ[e]) >= frac * (2ew.ρ[e] - x.σ[e]) || return false
    end
    return true
end

"""Largest t with z + tΔz ≥ 0 and y + tΔy ≥ 0."""
function _max_step(P::Problem, x::State, Δ::Vector{Float64})
    m, n = P.m, P.n
    tmax = Inf
    @inbounds for i in 1:m
        Δ[i] < 0 && (tmax = min(tmax, -x.z[i] / Δ[i]))
    end
    @inbounds for j in 1:n
        Δ[m+j] < 0 && (tmax = min(tmax, -x.y[j] / Δ[m+j]))
    end
    return tmax
end

"""
Backtracking line search from the fraction-to-boundary step: halve until the cones are
strictly feasible and, if λ > 1/4, Armijo holds for Φ_η. Returns the step (0 = failure) and
leaves the trial point in `xt`, `ewt`.
"""
function _line_search!(xt::State, ewt::EdgeWork, P::Problem, x::State, ew::EdgeWork,
                       Δ::Vector{Float64}, η::Float64, λ::Float64, o, prof)
    soc = o.soc
    function ok(t)                               # trial point strictly feasible
        prof[:n_ls_evals] += 1
        _set_trial!(xt, P, x, ew, Δ, t, soc)
        edge_terms!(ewt, P, xt) || return false
        o.cone_ftb || return true                # [dev] cone fraction-to-boundary
        _keeps_distance(P, x, ew, xt, ewt) && return true
        prof[:n_ftb_reject] += 1
        return false
    end
    (isfinite(λ) && all(isfinite, Δ)) || return 0.0
    tmax = _max_step(P, x, Δ)
    t = min(1.0, o.step_fraction * tmax)
    if λ <= 0.25                                  # quadratic convergence region: full step
        for _ in 1:60
            ok(t) && return t
            t /= 2
        end
        return 0.0
    end
    # Backtracking from t = 1 (Armijo).
    tA, dA = 0.0, Inf
    for _ in 1:60
        if ok(t)
            d = _phi_change(P, x, ew, xt, ewt, Δ, t, η)
            if d <= -0.01 * t * λ^2
                tA, dA = t, d
                break
            end
        end
        t /= 2
    end
    # [dev] o.damped: compare with the damped step 1/(1+λ)
    td = min(1 / (1 + λ), o.step_fraction * tmax)
    if o.damped && tA != td
        if ok(td)
            dd = _phi_change(P, x, ew, xt, ewt, Δ, td, η)
            if dd < dA                            # trial point already holds the damped step
                prof[:n_damped] += 1
                return td
            end
        end
    end
    tA == 0.0 && return 0.0
    _set_trial!(xt, P, x, ew, Δ, tA, soc)         # restore the Armijo point
    edge_terms!(ewt, P, xt)
    return tA
end
