module SparseBlockJacobians

export
    update_residuals!, update_jacobian!


@inline foldlargs(op, x) = x
@inline foldlargs(op, x1, x2, xs...) = foldlargs(op, op(x1, x2), xs...)

function update_residuals!(out::Array{Float64}, funcs)
    foldlargs(0, funcs...) do offset, (range, fun)
        @inbounds for i in 1:range
            out[i + offset] = fun(i)
        end
        offset + range
    end
    nothing
end

function update_jacobian!(out::Array{Float64}, funcs)
    foldlargs(0, funcs...) do offset, (range, block_size, fun)
        @inbounds for i in 1:block_size:(1+block_size*range)
            out[i + offset:i + offset+block_size] .= fun(i)
        end
        offset + range
    end
    nothing
end

end # module
