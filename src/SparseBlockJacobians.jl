module SparseBlockJacobians

export
    update_residuals!, update_jacobian_values!, update_jacobian_indices!

@inline foldlargs(op, x) = x
@inline foldlargs(op, x1, x2, xs...) = foldlargs(op, op(x1, x2), xs...)

function update_residuals!(out::Array{Float64}, funcs)
    foldlargs(0, funcs...) do offset, (range, fun)
        for i in 1:range
            out[i + offset] = fun(i)
        end
        offset + range
    end
    nothing
end

function update_jacobian_values!(out, funcs)
    foldlargs(0, funcs...) do offset, (range, block_size, fun)
        for i in 1:range
            out.nzval[1+block_size*i + offset:1+block_size*i + offset+block_size] .= fun(i)
        end
        offset + range
    end
    nothing
end

function update_jacobian_indices!(out, funcs)
    colacc = 1
    foldlargs(0, funcs...) do offset, (range, block_size, fun)
        for i in 0:(range-1)
            display("$i, $(fun(i))")
            out.rowval[1+block_size*i + offset:1+block_size*i + offset+block_size] .= fun(i)
            out.colptr[1+i + offset:1+i + offset+block_size] .= colacc
            colacc += block_size
        end
        offset + range
    end
    nothing
end

end # module
