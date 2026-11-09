using Test, Random, SparseArrays, LinearAlgebra
using ForwardDiff, Clarabel
include("reference.jl")
using MatrixNormIPM: EdgeWork, edge_terms!, gradient!, HessData, hessian_parts!, NewtonSystem,
                assemble!, newton_solve, decrement2, upper_bound, lower_bound, analytic_center

const TOL = 1e-8

@testset "MatrixNormIPM" begin

@testset "barrier derivatives vs ForwardDiff (p=$p, q=$q)" for (p, q) in [(3.0, 2.0), (2.0, 2.0), (4.0, 1.5)]
    rng = MersenneTwister(1)
    A = sprand(rng, 7, 5, 0.5) + sparse(1:5, 1:5, ones(5), 7, 5)
    P = Problem(A ./ maximum(nonzeros(A)), p, q)
    x = random_interior(P; rng)
    ew = EdgeWork(P)
    @test edge_terms!(ew, P, x)
    d0 = zeros(P.m + P.n + P.s)
    f(d) = barrier_generic(P, x, d)
    gF = ForwardDiff.gradient(f, d0)
    HF = ForwardDiff.hessian(f, d0)
    g = zeros(length(d0))
    gradient!(g, P, x, ew)
    @test g ≈ gF rtol = 1e-12
    gD, HD = dense_derivatives(P, x, Float64)
    @test HD ≈ HF rtol = 1e-12

    # Theorem 10: D (H_w - H_wγ H_γ⁻¹ H_wγᵀ) D = L  (equality-constrained version: no rank-2 term)
    N = P.m + P.n
    sys = NewtonSystem(P; solver=:cholesky)
    assemble!(sys, x, ew)
    Hw, Hwg, Hg = HF[1:N, 1:N], HF[1:N, N+1:end], HF[N+1:end, N+1:end]
    D = Diagonal(vcat(x.z, x.y))
    @test D * (Hw - Hwg * (Hg \ Hwg')) * D ≈ Matrix(sys.L) rtol = 1e-10
    @test sys.hd.hγ ≈ diag(Hg) rtol = 1e-12
end

@testset "Newton solve vs dense KKT ($solver, cheb_steps=$k)" for (solver, k) in
        [(:cholesky, 0), (:approxchol, 0), (:approxchol, 2), (:approxchol, 6)]
    rng = MersenneTwister(2)
    A = sprand(rng, 40, 30, 0.15) + sparse(1:30, 1:30, ones(30), 40, 30)
    P = Problem(A ./ maximum(nonzeros(A)), 3.0, 2.0)
    x = random_interior(P; rng)
    ew = EdgeWork(P)
    edge_terms!(ew, P, x)
    g, H = dense_derivatives(P, x, Float64)
    b = randn(rng, length(g))
    Δref = dense_kkt_solve(P, H, b)
    sys = NewtonSystem(P; solver, cheb_steps=k)
    assemble!(sys, x, ew)
    Δ, u = newton_solve(sys, x, b)
    Hn(v) = sqrt(dot(v, H * v))
    # the error of the Newton step is that of the inner operator (Lemma 11): ≤ δ, which is
    # exact for CHOLMOD, ≲ 0.5 for one approxchol application, and shrinks with Chebyshev steps
    errtol = solver === :cholesky ? 1e-8 : (k == 0 ? 0.5 : (k == 2 ? 0.1 : 1e-2))
    @test Hn(Δ - Δref) <= errtol * Hn(Δref)
    # constraints 1ᵀΔz = 1ᵀΔy = 0 hold up to rounding
    @test abs(sum(Δ[1:P.m])) ≤ 1e-9 * sum(abs, Δ[1:P.m])
    @test abs(sum(Δ[P.m+1:P.m+P.n])) ≤ 1e-9 * sum(abs, Δ[P.m+1:P.m+P.n])
    @test sqrt(decrement2(sys, u, b[P.m+P.n+1:end])) ≈ Hn(Δ) rtol = 1e-8
end

@testset "approxchol inner operator is fixed, symmetric, and improves with Chebyshev steps (cheb_steps=$k)" for k in (0, 1, 3)
    rng = MersenneTwister(8)
    n = 60
    A = sprand(rng, n, n, 4 / n) + sparse(1:n, 1:n, rand(rng, n))
    P = Problem(A ./ maximum(nonzeros(A)), 3.0, 2.0)
    deep = Ref{Any}(nothing)
    pqnorm(A, 3.0, 2.0; tol=1e-12, max_iter=15, solver=:cholesky, callback=info -> (deep[] = copy(info.x)))
    x = deep[]
    ew = EdgeWork(P); edge_terms!(ew, P, x)
    sys = NewtonSystem(P; solver=:approxchol, cheb_steps=k)
    assemble!(sys, x, ew)
    N = sys.N
    Pm = reduce(hcat, [sys.lsolve(Vector{Float64}(I[1:N, j])) for j in 1:N])
    @test norm(Pm - Pm') ≤ 1e-12 * norm(Pm)                   # fixed symmetric linear operator
    b = randn(rng, N)
    @test sys.lsolve(b) ≈ Pm * b rtol = 1e-12                  # same operator for every b
    C = cholesky(Symmetric(Matrix(sys.L))).U
    ev = eigvals(Symmetric(C * ((Pm + Pm') / 2) * C'))
    δ = maximum(abs.(ev .- 1))                               # (1-δ)L⁻¹ ⪯ P ⪯ (1+δ)L⁻¹
    @test sys.cheb_k == k + 1
    @test minimum(ev) > 0                                    # P is positive definite
    if k == 0
        @test δ < 0.6
    else
        # the interval comes from 6 Lanczos steps (an estimate), so δ_k is approximate
        @test δ ≤ 2 * sys.delta_est + 0.05
        @test δ < (k == 1 ? 0.3 : 0.05)
    end
end

@testset "disconnected A (block diagonal, p=$p, q=$q)" for (p, q) in [(2.0, 2.0), (1.5, 1.5), (3.0, 2.0)]
    rng = MersenneTwister(9)
    A1 = sprand(rng, 30, 20, 0.2) + sparse(1:20, 1:20, ones(20), 30, 20)
    A2 = 0.9 .* (sprand(rng, 25, 25, 0.2) + I)
    A = blockdiag(sparse(A1), sparse(A2))
    r = pqnorm(A, p, q; tol=TOL, solver=:cholesky)
    ref = jump_reference(A, p, q; optimizer=Clarabel.Optimizer)
    @test r.status == :optimal
    @test abs(r.value - ref) ≤ 1e-6 * ref
    if p == q                                                # ‖A‖ = max over blocks
        r1, r2 = pqnorm(A1, p, q; tol=TOL, solver=:cholesky), pqnorm(A2, p, q; tol=TOL, solver=:cholesky)
        @test abs(r.value - max(r1.value, r2.value)) ≤ 2TOL * r.value
    end
    # degenerate maximizer: the adaptive inner accuracy (development option) recovers approxchol
    ra = pqnorm(A, p, q; tol=TOL, opts=(adaptive_cheb=true,))
    @test ra.status == :optimal
    @test abs(ra.value - ref) ≤ 1e-6 * ref
end

@testset "closed forms and special cases" begin
    rng = MersenneTwister(3)
    A = sprand(rng, 30, 20, 0.2)
    M = Matrix(A)
    r = pqnorm(A, 2, 2; tol=TOL, solver=:cholesky)
    @test r.status == :optimal
    @test r.value ≤ opnorm(M) * (1 + 1e-12)
    @test r.upper ≥ opnorm(M) * (1 - 1e-12)
    @test abs(r.value - opnorm(M)) ≤ TOL * opnorm(M)
    @test norm(r.u, 2) ≈ 1
    @test norm(A * r.u, 2) ≈ r.value

    @test pqnorm(A, 1, 1).value ≈ opnorm(M, 1)               # max column sum
    @test pqnorm(A, Inf, Inf).value ≈ opnorm(M, Inf)         # max row sum
    r = pqnorm(A, 3, 1)                                     # ‖Aᵀ1‖_{p*}
    @test r.value ≈ norm(M' * ones(30), 1.5)
    @test norm(A * r.u, 1) ≈ r.value
    @test pqnorm(A, Inf, 2).value ≈ norm(M * ones(20), 2)
    @test pqnorm(spzeros(4, 3), 3, 2).value == 0

    # zero rows / columns are handled (u is padded with zeros)
    B = sprand(rng, 25, 15, 0.2)
    B[3, :] .= 0
    B[:, 4] .= 0
    dropzeros!(B)
    r = pqnorm(B, 2, 2; tol=TOL, solver=:cholesky)
    @test r.status == :optimal
    @test abs(r.value - opnorm(Matrix(B))) ≤ TOL * opnorm(Matrix(B))
    @test length(r.u) == 15 && r.u[4] == 0

    @test_throws ArgumentError pqnorm(A, 1.5, 2)
    @test_throws ArgumentError pqnorm(-A, 2, 2)
    for kw in ((tol=0.0,), (solver=:foo,), (cheb_steps=-1,), (split=-1,), (step_fraction=1.0,),
               (centrality=0.0,), (barrier_growth=1.0,), (max_iter=-1,),
               (opts=(foo=1,),))
        @test_throws ArgumentError pqnorm(A, 3, 2; kw...)
    end
    # scaling invariance
    @test pqnorm(7.5 .* A, 3, 2; tol=TOL, solver=:cholesky).value ≈
          7.5 * pqnorm(A, 3, 2; tol=TOL, solver=:cholesky).value rtol = 1e-7
end

@testset "API: p-only wrapper, limits, statuses" begin
    rng = MersenneTwister(10)
    A = sprand(rng, 40, 30, 0.15) + sparse(1:30, 1:30, ones(30), 40, 30)
    r1 = pqnorm(A, 3; tol=1e-7, solver=:cholesky)            # deterministic backend
    r2 = pqnorm(A, 3, 3; tol=1e-7, solver=:cholesky)
    @test r1.value == r2.value && r1.upper == r2.upper && r1.iterations == r2.iterations
    @test pqnorm(A, 3).status == :optimal                    # default: approxchol, q = p
    r = pqnorm(A, 3, 2; tol=1e-12, max_iter=3)
    @test r.status == :max_iter && r.iterations == 3 && r.value ≤ r.upper
    r = pqnorm(A, 3, 2; tol=1e-12, time_limit=0.0)
    @test r.status == :time_limit && r.iterations == 1
    @test pqnorm(A, 3, 2; cheb_steps=2).status == :optimal
    @test pqnorm(A, 3, 2; split=2).status == :optimal
end

@testset "vs JuMP + Clarabel ($solver, p=$p, q=$q)" for solver in (:cholesky, :approxchol),
        (p, q) in [(3.0, 2.0), (4.0, 1.5), (2.0, 1.2), (1.5, 1.5), (10.0, 1.1)]
    rng = MersenneTwister(4)
    A = sprand(rng, 40, 30, 0.15) + sparse(1:30, 1:30, rand(rng, 30), 40, 30)
    r = pqnorm(A, p, q; tol=TOL, solver)
    ref = jump_reference(A, p, q; optimizer=Clarabel.Optimizer)
    @test r.status == :optimal
    @test r.gap ≤ TOL
    @test r.value ≤ r.upper
    @test abs(r.value - ref) ≤ 1e-6 * ref                    # Clarabel's own accuracy is ~1e-8
    @test norm(r.u, p) ≈ 1
    @test norm(A * r.u, q) ≈ r.value rtol = 1e-12            # value is attained by u
end

@testset "certificates are valid along the path" begin
    rng = MersenneTwister(5)
    A = sprand(rng, 30, 25, 0.2) + sparse(1:25, 1:25, ones(25), 30, 25)
    opt = opnorm(Matrix(A))
    viol = Ref(0)
    cb = info -> begin
        s = maximum(nonzeros(A))
        lb, _ = lower_bound(info.P, info.x)
        ub = upper_bound(info.P, info.x, info.ew)
        ub2 = upper_bound(info.P, info.x, info.ew, info.Δ)
        (lb * s ≤ opt * (1 + 1e-12) && ub * s ≥ opt * (1 - 1e-12) && ub2 * s ≥ opt * (1 - 1e-12)) || (viol[] += 1)
    end
    r = pqnorm(A, 2, 2; tol=1e-10, solver=:cholesky, callback=cb)
    @test viol[] == 0
    @test r.status == :optimal
end

@testset "backends and inner accuracies agree ($solver, cheb_steps=$k, split=$sp)" for (solver, k, sp) in
        [(:cholesky, 0, 0), (:approxchol, 0, 0), (:approxchol, 1, 0), (:approxchol, 4, 0),
         (:approxchol, 0, 4)]
    rng = MersenneTwister(6)
    A = sprand(rng, 80, 60, 0.06) + sparse(1:60, 1:60, rand(rng, 60), 80, 60)
    ref = pqnorm(A, 3, 2; tol=1e-10, solver=:cholesky)
    r = pqnorm(A, 3, 2; tol=1e-6, solver, cheb_steps=k, split=sp, max_iter=500)
    @test r.status == :optimal
    @test r.value ≤ ref.upper && r.upper ≥ ref.value          # certificates bracket the truth
    @test abs(r.value - ref.value) ≤ 1e-6 * ref.value
end

end
