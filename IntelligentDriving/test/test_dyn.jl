using BenchmarkTools

include(joinpath(@__DIR__, "../src/dyn.jl"))
include(joinpath(@__DIR__, "../src/diff.jl"))

xdim, udim, N = 4, 2, 100
X, U, P = randn(xdim, N), randn(udim, N), vcat(0.1 * ones(N)', ones(2, N))


#stack(x_list) =
#  reshape(reduce(hcat, map(x -> view(x, :), x_list)), size(x_list[1])..., :)

@btime begin
  f = mapreduce(i -> unicycle_f(X[:, i], U[:, i], P[:, i]), hcat, 1:N)
  #fx = stack(map(i -> unicycle_fx(X[:, i], U[:, i], P[:, i]), 1:N); dims = 3)
  #fu = stack(map(i -> unicycle_fu(X[:, i], U[:, i], P[:, i]), 1:N); dims = 3)
end

@btime begin
  f = mapreduce(
    i -> unicycle_f(X[:, i], U[:, i], P[:, i]),
    (a, b) -> cat(a, b; dims = 2),
    1:N,
  )
  #fx = stack(map(i -> unicycle_fx(X[:, i], U[:, i], P[:, i]), 1:N); dims = 3)
  #fu = stack(map(i -> unicycle_fu(X[:, i], U[:, i], P[:, i]), 1:N); dims = 3)
end

@btime begin
  f = reduce(
    (a, b) -> cat(a, b; dims = 2),
    map(i -> unicycle_f(X[:, i], U[:, i], P[:, i]), 1:N),
  )
  #fx = stack(map(i -> unicycle_fx(X[:, i], U[:, i], P[:, i]), 1:N); dims = 3)
  #fu = stack(map(i -> unicycle_fu(X[:, i], U[:, i], P[:, i]), 1:N); dims = 3)
end

@btime begin
  f = reduce(
    hcat,
    map(
      x -> reshape(x, :),
      map(i -> unicycle_f(X[:, i], U[:, i], P[:, i]), 1:N),
    ),
  )
  #fx = stack(map(i -> unicycle_fx(X[:, i], U[:, i], P[:, i]), 1:N); dims = 3)
  #fu = stack(map(i -> unicycle_fu(X[:, i], U[:, i], P[:, i]), 1:N); dims = 3)
end

#@btime begin
#  fx_ = stack(
#    map(i -> jacobian_gen(x -> unicycle_f(x, U[:, i], P[:, i]))(X[:, i]), 1:N);
#    dims = 3,
#  )
#  fu_ = stack(
#    map(i -> jacobian_gen(u -> unicycle_f(X[:, i], u, P[:, i]))(U[:, i]), 1:N);
#    dims = 3,
#  )
#end

return
