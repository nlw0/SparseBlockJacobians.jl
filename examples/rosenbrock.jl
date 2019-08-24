using SparseArrays

using Revise
using SparseBlockJacobians

function myresidue!(out, x)
    N = length(x)
    differences(n) = 10.0 * (x[n+1] - x[n]^2)
    residue(n) = 1.0 - x[n]
    blocks = ((N-1, differences), (N-1, residue))
    calculate_model!(out, blocks)
    nothing
end

function myjaco!(J, x)
    N = length(x)

    differences(n) = (10.0, - 20.0 * x[n])
    residue(n) = (-1.0,)

    blocks = ((N-1, 2, differences), (N-1, 1, residue))
    update_jacobian_values!(J, blocks)
    nothing
end

function myjacoini!(J, x)
    N = length(x)

    differences(n) = (n+1, n)
    residue(n) = (n,)

    blocks = ((N-1, 2, differences), (N-1, 1, residue))
    update_jacobian_indices!(J, blocks)
    nothing
end

N = 5
cols = 2*(N-1)
vals = 3*(N-1)

aa = SparseMatrixCSC(
    5,
    cols,
    collect(1:cols+1),
    Array{Int64}(undef,vals),
    Array{Float64}(undef,vals)
)

x = rand(5)

myjacoini!(aa, x)
myjaco!(aa, x)

# xx = sparse([
#     1 1 0 0 0
#     0 1 1 0 0
#     0 0 1 0 0
#     1 1 1 1 1
# ])
