# An Interior Point Method for computing the $p \to q$ norm of a non-negative matrix

This repository contains code to efficiently compute the $p \to q$ norm of a non-negative matrix $A$, when $p \ge q$, which is based on the following [paper](https://arxiv.org/abs/2611.09601):

```bibtex
@misc{pirazzini2026matrixnormipm,
      title={A Fast and Accurate Algorithm for the $p \to q$ Norm of Sparse Non-Negative Matrices}, 
      author={Marco Pirazzini and Daniel A. Spielman},
      year={2026},
      eprint={2611.09601},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      url={https://arxiv.org/abs/2611.09601},
}
```

## Installation

Requires [Julia](https://julialang.org/downloads/) ≥ 1.10.

```bash
git clone https://github.com/marcopirazzini/MatrixNormIPM.jl
cd MatrixNormIPM.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

Start Julia with `julia --project=.` and run:

```julia
using MatrixNormIPM, SparseArrays

A = sprand(1000, 500, 0.01)        # sparse, entrywise non-negative
r = pqnorm(A, 3, 2; tol=1e-6)      # ‖A‖_{3→2}, requires p ≥ q ≥ 1

r.value    # certified lower bound on ‖A‖_{p→q}
r.upper    # certified upper bound; (upper - value)/value ≤ tol
r.u        # maximizer, ‖u‖_p = 1
r.status   # :optimal on success
```

`pqnorm(A, p)` computes ‖A‖_{p→p}. By default each Newton step is solved with the
approximate Laplacian solver from [Laplacians.jl](https://github.com/danspielman/Laplacians.jl).
Pass `solver=:cholesky` to use an exact sparse Cholesky instead. See `?pqnorm` for all options.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```