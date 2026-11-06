# Newton systems on {Σz = 1, Σy = 1} (Alg. 3/4 of the paper):
# 1. Eliminate the diagonal γ-block (Schur complement).
# 2. In scaled variables Δw = D u, D = Diag(z, y), solve the KKT system
#        [L E; Eᵀ 0] [u; λ] = [D b̃; 0],   E = [z 0; 0 y],   L = D M D strictly SDD.
# 3. Apply a fixed linear operator P ≈ L⁻¹: CHOLMOD, one approxchol application P₀, or
#    Chebyshev iterations preconditioned by P₀ (Theorem 2).
# 4. Constraint preconditioner Π(r) = P r - W G⁻¹ Eᵀ(P r), W = P E, G = EᵀW, which maps into
#    null(Eᵀ); the Newton step is one application of Π per right-hand side (Lemma 11).

Base.@kwdef mutable struct NewtonSystem
    P::Problem
    N::Int                                   # m + n
    hd::HessData
    L::SparseMatrixCSC{Float64,Int}          # full symmetric storage, fixed pattern
    pos_off1::Vector{Int}                    # nzval index of L[i, m+j] for edge e
    pos_off2::Vector{Int}                    # nzval index of L[m+j, i]
    pos_diag::Vector{Int}                    # nzval index of L[v, v]
    e1::Vector{Float64} = zeros(N)           # [z; 0]
    e2::Vector{Float64} = zeros(N)           # [0; y]
    W1::Vector{Float64} = zeros(N)           # lsolve(e1)
    W2::Vector{Float64} = zeros(N)           # lsolve(e2)
    Ginv::Matrix{Float64} = zeros(2, 2)      # (EᵀW)⁻¹
    # inner solver
    solver::Symbol                           # :cholesky | :approxchol
    cheb_steps::Int
    split::Int                               # approxchol edge splitting
    lanczos_steps::Int
    lanczos_every::Int = 1                   # [dev]
    chol::Any = nothing                      # CHOLMOD factor (symbolic factorization reused)
    lsolve::Any = nothing                    # inner operator b -> P b
    n_builds::Int = 0
    bounds::Tuple{Float64,Float64} = (NaN, NaN)   # estimated spectrum of P₀L
    # diagnostics of the last Hessian (see IterStats) and counters reset per Newton step
    cheb_k::Int = 0
    kappa::Float64 = NaN
    delta_est::Float64 = NaN
    inner_calls::Int = 0
    inner_its::Int = 0
    t_build::Float64 = 0.0
    t_solve::Float64 = 0.0
    t_factor::Float64 = 0.0
    t_lanczos::Float64 = 0.0
    t_apply::Float64 = 0.0
end

function _nzindex(A::SparseMatrixCSC, r::Int, c::Int)
    rng = A.colptr[c]:(A.colptr[c+1]-1)
    k = searchsortedfirst(view(A.rowval, rng), r)
    @assert k <= length(rng) && A.rowval[rng[k]] == r
    return rng[k]
end

function NewtonSystem(P::Problem; solver::Symbol=:approxchol, cheb_steps::Integer=0,
                      split::Integer=0, lanczos_steps::Integer=6, lanczos_every::Integer=1)
    lanczos_steps >= 2 || throw(ArgumentError("lanczos_steps must be ≥ 2, got $lanczos_steps"))
    m, n, s = P.m, P.n, P.s
    N = m + n
    I = vcat(1:N, P.ei, m .+ P.ej)
    J = vcat(1:N, m .+ P.ej, P.ei)
    L = sparse(I, J, ones(N + 2s), N, N)
    pos_diag = [_nzindex(L, v, v) for v in 1:N]
    pos_off1 = [_nzindex(L, P.ei[e], m + P.ej[e]) for e in 1:s]
    pos_off2 = [_nzindex(L, m + P.ej[e], P.ei[e]) for e in 1:s]
    return NewtonSystem(; P, N, hd=HessData(P), L, pos_off1, pos_off2, pos_diag,
                        solver, cheb_steps, split, lanczos_steps, lanczos_every)
end

function reset_counters!(sys::NewtonSystem)
    sys.inner_calls = 0
    sys.inner_its = 0
    sys.t_build = 0.0
    sys.t_solve = 0.0
    sys.t_factor = 0.0
    sys.t_lanczos = 0.0
    sys.t_apply = 0.0
end

# ---------------------------------------------------------------------------------------
# Inner Laplacian solvers

"""
Adjacency of L grounded at vertex N+1 (joined to each vertex by its diagonal excess), built
from the exact weights: recomputing the excess as L*1 cancels catastrophically near the optimum.
"""
function grounded_adjacency(sys::NewtonSystem)
    P, hd, N, m = sys.P, sys.hd, sys.N, sys.P.m
    keep = findall(>(0.0), hd.r1)
    I = vcat(P.ei[keep], m .+ P.ej[keep], 1:N, fill(N + 1, N))
    J = vcat(m .+ P.ej[keep], P.ei[keep], fill(N + 1, N), 1:N)
    V = vcat(hd.r1[keep], hd.r1[keep], hd.exc, hd.exc)
    return sparse(I, J, V, N + 1, N + 1)
end

@inline function _ground(xa::Vector{Float64}, N::Int)
    x = Vector{Float64}(undef, N)
    c = xa[N+1]
    @inbounds for v in 1:N
        x[v] = xa[v] - c
    end
    return x
end

function build_inner!(sys::NewtonSystem)
    N = sys.N
    t0 = time()
    sys.n_builds += 1
    if sys.solver === :cholesky
        try
            if sys.chol === nothing
                sys.chol = cholesky(Symmetric(sys.L))
            else
                cholesky!(sys.chol, Symmetric(sys.L))
            end
        catch err
            err isa PosDefException || rethrow()
            # near the optimum L is numerically singular; a tiny shift keeps the factor usable
            shift = 1e-13 * maximum(view(sys.L.nzval, sys.pos_diag))
            sys.chol = cholesky(Symmetric(sys.L); shift=shift)
        end
        sys.t_factor += time() - t0
        F = sys.chol
        op = b -> (sys.inner_its += 1; F \ b)
    else
        a1 = grounded_adjacency(sys)
        ldli = sys.split >= 1 ?
            Laplacians.approxChol(Laplacians.LLmatp(a1, sys.split), sys.split, sys.split) :
            Laplacians.approxChol(Laplacians.LLmatp(a1))
        sys.t_factor += time() - t0
        P0 = function (b)                          # one application of the approxchol factor
            sys.inner_its += 1
            return _ground(Laplacians.LDLsolver(ldli, vcat(b, -sum(b))), N)
        end
        if sys.cheb_steps == 0
            sys.cheb_k, sys.kappa, sys.delta_est = 1, NaN, NaN
            op = P0
        else
            # Theorem 2: k applications of P₀ give spectrum of P·L in [1-δ_k, 1+δ_k]
            if isnan(sys.bounds[1]) || (sys.n_builds - 1) % sys.lanczos_every == 0
                t1 = time()
                sys.bounds = _lanczos_bounds(sys.L, P0, N; steps=sys.lanczos_steps)
                sys.t_lanczos += time() - t1
            end
            lo, hi = sys.bounds
            k = sys.cheb_steps + 1
            sys.cheb_k, sys.kappa = k, hi / lo
            sys.delta_est = 1 / cosh(k * acosh((hi + lo) / (hi - lo)))
            L = sys.L
            op = b -> _chebyshev(L, P0, b, lo, hi, k)
        end
    end
    sys.lsolve = _counted(sys, op)
    return sys
end

"""Count and time every application of the inner operator `op`."""
function _counted(sys::NewtonSystem, op)
    return function (b)
        t1 = time()
        sys.inner_calls += 1
        x = op(b)
        sys.t_apply += time() - t1
        return x
    end
end

"""
Extreme eigenvalues of P₀L from `steps` PCG iterations (Lanczos) on a random right-hand side,
padded outward by `pad` since Ritz values lie inside the spectrum.
"""
function _lanczos_bounds(L, P0, N; steps::Int=6, pad::Float64=0.1)
    b = randn(N)
    x = zeros(N)
    r = copy(b)
    z = P0(r)
    p = copy(z)
    rz = dot(r, z)
    αs = Float64[]
    βs = Float64[]
    Ap = similar(b)
    for _ in 1:steps
        mul!(Ap, L, p)
        α = rz / dot(p, Ap)
        push!(αs, α)
        @. x += α * p
        @. r -= α * Ap
        z = P0(r)
        rznew = dot(r, z)
        β = rznew / rz
        (isfinite(β) && β > 0) || break
        push!(βs, β)
        rz = rznew
        @. p = z + β * p
    end
    k = length(αs)
    d = [1 / αs[i] + (i > 1 ? βs[i-1] / αs[i-1] : 0.0) for i in 1:k]
    e = [sqrt(βs[i]) / αs[i] for i in 1:k-1]
    ev = eigvals(SymTridiagonal(d, e))
    lo, hi = minimum(ev), maximum(ev)
    # lo can come out ≤ 0 near the optimum; clamping it keeps P positive semidefinite
    lo = max(lo, hi / 100)
    return lo / (1 + pad), hi * (1 + pad)
end

"""
k steps of Chebyshev iteration for L x = b preconditioned by P₀ on [lo, hi], from x₀ = 0
(Saad, Alg. 12.1): the fixed symmetric operator q_k(P₀L)P₀.
"""
function _chebyshev(L, P0, b::Vector{Float64}, lo::Float64, hi::Float64, k::Int)
    θ = (hi + lo) / 2
    δc = (hi - lo) / 2
    σ1 = θ / δc
    ρ = 1 / σ1
    x = zeros(length(b))
    r = copy(b)
    d = P0(r) ./ θ
    Ld = similar(b)
    for it in 1:k
        @. x += d
        it == k && break
        mul!(Ld, L, d)
        @. r -= Ld
        ρnew = 1 / (2σ1 - ρ)
        z = P0(r)
        @. d = ρnew * ρ * d + (2ρnew / δc) * z
        ρ = ρnew
    end
    return x
end

# ---------------------------------------------------------------------------------------
# Assembly and the constraint preconditioner

"""Assemble L, E and the inner solver at `x`; precompute W = P E and G = EᵀW."""
function assemble!(sys::NewtonSystem, x::State, ew::EdgeWork)
    t0 = time()
    P, hd, m, N = sys.P, sys.hd, sys.P.m, sys.N
    hessian_parts!(hd, P, x, ew)
    nz = sys.L.nzval
    @inbounds begin
        for v in 1:N
            nz[sys.pos_diag[v]] = hd.exc[v]
        end
        for e in 1:P.s
            r1 = hd.r1[e]
            nz[sys.pos_off1[e]] = -r1
            nz[sys.pos_off2[e]] = -r1
            nz[sys.pos_diag[P.ei[e]]] += r1
            nz[sys.pos_diag[m+P.ej[e]]] += r1
        end
    end
    fill!(sys.e1, 0.0)
    fill!(sys.e2, 0.0)
    sys.e1[1:m] .= x.z
    sys.e2[m+1:N] .= x.y
    build_inner!(sys)
    sys.W1 = sys.lsolve(sys.e1)
    sys.W2 = sys.lsolve(sys.e2)
    g11 = dot(sys.e1, sys.W1)
    g22 = dot(sys.e2, sys.W2)
    g12 = (dot(sys.e1, sys.W2) + dot(sys.e2, sys.W1)) / 2
    det = g11 * g22 - g12 * g12
    sys.Ginv .= [g22 -g12; -g12 g11] ./ det
    sys.t_build += time() - t0
    return sys
end

"""v ← v - E(EᵀE)⁻¹Eᵀv: changes only the multiplier, but removes the huge O(η) part of v."""
function _deflate!(sys::NewtonSystem, v::Vector{Float64})
    c1 = dot(sys.e1, v) / dot(sys.e1, sys.e1)
    c2 = dot(sys.e2, v) / dot(sys.e2, sys.e2)
    @inbounds for k in eachindex(v)
        v[k] -= c1 * sys.e1[k] + c2 * sys.e2[k]
    end
    return v
end

"""x ← x - W G⁻¹ Eᵀx: restores Eᵀx = 0."""
function _reproject!(sys::NewtonSystem, x::Vector{Float64})
    c1 = dot(sys.e1, x)
    c2 = dot(sys.e2, x)
    λ1 = sys.Ginv[1, 1] * c1 + sys.Ginv[1, 2] * c2
    λ2 = sys.Ginv[2, 1] * c1 + sys.Ginv[2, 2] * c2
    @inbounds for v in eachindex(x)
        x[v] -= λ1 * sys.W1[v] + λ2 * sys.W2[v]
    end
    return x
end

"""
Π(r) = P r - W G⁻¹ Eᵀ(P r), so EᵀΠ(r) = 0. For stability r is deflated first and the result
re-projected once more (Gould, Hribar & Nocedal 2001).
"""
function project_precond(sys::NewtonSystem, r::Vector{Float64})
    x = sys.lsolve(_deflate!(sys, copy(r)))
    _reproject!(sys, x)
    return _reproject!(sys, x)
end

"""
Newton step for right-hand side `b = [b_z; b_y; b_γ]` at `x` (after `assemble!`). Returns
`(Δ, u)` with `u = D⁻¹Δ_w` the scaled reduced solution.
"""
function newton_solve(sys::NewtonSystem, x::State, b::Vector{Float64})
    t0 = time()
    P, hd, m, n, N = sys.P, sys.hd, sys.P.m, sys.P.n, sys.N
    rhs = b[1:N]                                  # b̃_w = b_w - H_wγ H_γ⁻¹ b_γ
    @inbounds for e in 1:P.s
        bγ = b[N+e]
        rhs[P.ei[e]] -= hd.cz[e] * bγ
        rhs[m+P.ej[e]] -= hd.cy[e] * bγ
    end
    @inbounds for i in 1:m                        # scale by D
        rhs[i] *= x.z[i]
    end
    @inbounds for j in 1:n
        rhs[m+j] *= x.y[j]
    end
    _deflate!(sys, rhs)
    u = project_precond(sys, rhs)                 # Alg. 3: one application of Π
    Δ = Vector{Float64}(undef, N + P.s)
    @inbounds begin
        for i in 1:m
            Δ[i] = x.z[i] * u[i]
        end
        for j in 1:n
            Δ[m+j] = x.y[j] * u[m+j]
        end
        for e in 1:P.s                            # Δγ = H_γ⁻¹(b_γ - H_wγᵀ Δw)
            Δ[N+e] = b[N+e] / hd.hγ[e] - hd.cz[e] * Δ[P.ei[e]] - hd.cy[e] * Δ[m+P.ej[e]]
        end
    end
    sys.t_solve += time() - t0
    return Δ, u
end

"""
‖Δ‖²_H = uᵀLu + Σ_e bγ_e²/H_γ(e) (Lemma 11): a sum of non-negative terms, so no cancellation
for large η.
"""
function decrement2(sys::NewtonSystem, u::Vector{Float64}, bγ::AbstractVector{Float64})
    P, hd, m = sys.P, sys.hd, sys.P.m
    acc = 0.0
    @inbounds begin
        for e in 1:P.s
            d = u[P.ei[e]] - u[m+P.ej[e]]
            acc += hd.r1[e] * d * d + bγ[e]^2 / hd.hγ[e]
        end
        for v in 1:sys.N
            acc += hd.exc[v] * u[v]^2
        end
    end
    return acc
end
