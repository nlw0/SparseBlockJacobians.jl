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
    foldlargs(0, funcs...) do val_offset, (range, block_size, fun)
        for i in 0:(range-1)
            out.nzval[1 + val_offset + block_size*i : 1 + val_offset + block_size*i + block_size-1] .= fun(i+1)
        end
        val_offset + range * block_size
    end
    nothing
end

function update_jacobian_indices!(out, funcs)
    out.colptr[1] = 1

    colacc = 1
    foldlargs((0,0), funcs...) do (col_offset, row_offset), (range, block_size, fun)
        for i in 0:(range-1)
            out.rowval[1 + row_offset + block_size*i : 1 + row_offset + block_size*i+ block_size-1] .= fun(i+1)
            colacc += block_size
            out.colptr[2+i+col_offset] = colacc
        end
        (col_offset + range, row_offset + range * block_size)
    end
    nothing
end

end # module
