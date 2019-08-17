using ForwardDiff
using LeastSquaresOptim
using Printf
using Plots
pyplot()


@inline foldlargs(op, x) = x
@inline foldlargs(op, x1, x2, xs...) = foldlargs(op, op(x1, x2), xs...)

function calculate_model!(out::Array{Float64}, funcs)
    foldlargs(0, funcs...) do offset, (range, fun)
        @inbounds for i in 1:range
            out[i + offset] = fun(i)
        end
        offset + range
    end
    nothing
end

function myresidue!(out, x, ref, Γ)
    sΓ = sqrt(Γ)
    N = length(x)
    regularization(n) = sΓ * (x[n+1] - x[n])
    fitness(n) = x[n] - ref[n]
    blocks = ((N-1, regularization), (N, fitness))
    calculate_model!(out, blocks)
    nothing
end

function myresidueparts(x, ref)
    out = zeros(length(x)*2-1)

    myresidue!(out, x, ref, 1.0)

    Hooke = sum(abs2,out[1:length(x)-1])
    Δ = sum(abs2, out[length(x):2*length(x)-1])

    Δ, Hooke
end

Npt = 16

data = rand(Npt) * 2
data .-= sum(data)/Npt

aa=[]
lab=[]
pp=[]

solutions = []

for n in -9:7*8
    myΓ= 1e-2 * 10.0^((n-1)*9/(9*8))
    myobjfun!(out, x) = myresidue!(out, x, data, myΓ)

    x0 = zeros(Npt)
    sol = optimize!(LeastSquaresProblem(x = x0, f! = myobjfun!, output_length=(Npt*2-1), autodiff=:central))

    push!(solutions, (n, myΓ, sol.minimizer))
end

for (n, myΓ, xo) in solutions
    Δ, Hooke = myresidueparts(xo, data)
    push!(aa, (Δ, Hooke))

    if n % 8 == 1 && n < 40
        push!(lab, (Δ, Hooke, @sprintf("Γ=%.1E", myΓ)))
    end
    if n % 8 == 1 && n<45
        plot(title=@sprintf("Γ=%.1E", myΓ))
        plot!(data, l=2, label="", style=:dash, color=:black)
        myp = plot!(xo, l=2, m=3, label="")
        push!(pp, myp)
    end
end

xx = hcat([[a...] for a in aa]...)

p1 = plot(pp..., layout=(3,2))
p2 = plot(xx[1,:], xx[2,:], l=2, m=2, size=(800*1.5, 450*1.5), ylabel="Smoothness error", xlabel="Position error", label="L-curve", title="Regularization example")
for (x,y,ll) in lab
    global p2 = annotate!(x,y,text(ll, 10, :left, :bottom))
end

plot(p1, p2, layout=2)
