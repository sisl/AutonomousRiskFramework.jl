### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ d7f2c6b4-1d37-11eb-24aa-2b2499d39686
try
	using Pkg
	Pkg.activate("../STLCG.jl/.")
	using AddPackage
catch
	Pkg.add("AddPackage")
	using AddPackage
end

# ╔═╡ c62ba2e2-1d38-11eb-0ab4-3b41bf022c00
@add using Revise, STLCG, EllipsisNotation, Plots

# ╔═╡ b1b1f35c-1d38-11eb-0e94-4f25525b267e
begin
	# setting up the signals
	# Using very easy signals so we can compare with the correct answer
	bs = 2
	t = 15
	x_dim = 3
	dim = 2
	x = Float32.(repeat(reshape(collect(1:t), 1, t, 1), bs, 1, x_dim))
	y = Float32.(repeat(reshape(collect(1:t), 1, t, 1), bs, 1, x_dim)) .+ 2

	plot(x[1,:,1])
	plot!(y[1,:,1])
end

# ╔═╡ 88c3412a-1d3e-11eb-2e19-9b70720f9ee6
x

# ╔═╡ eb614626-1d3c-11eb-351b-a5765ae4513d
# some settings
# parameters: these are the default values, but feel free to change them here to test it out
begin
	pscale=1    # scale for LessThan, GreaterThan, Equal
	scale=0     # scale for the minish and maxish function used in temporal operators, implies, and, or
	keepdims=true      # keep original dimension (should pretty much always be true)
	distributed=false  # if there are multiple indices that have the same max/min values, then mean over those to the gradient flows through all those values
end

# ╔═╡ c3e56212-1d3c-11eb-2d86-a9c88a21ca79
# Less than
lt = LessThan(:x, 0.0)

# ╔═╡ e3eaf676-1d3c-11eb-3439-ef7552a13c5e
ρt(lt, x; pscale, scale, keepdims, distributed), ρ(lt, x; pscale, scale, keepdims, distributed)

# ╔═╡ 754e9fb6-1d3e-11eb-1e32-4562174f72bb
# Greater than
gt = GreaterThan(:x, 0.0)

# ╔═╡ 7533e916-1d3e-11eb-2dd1-553a35cbeb07
ρt(gt, x; pscale, scale, keepdims, distributed), ρ(gt, x; pscale, scale, keepdims, distributed)

# ╔═╡ 7518b290-1d3e-11eb-22b4-693ab29c73b7
# Negation
not = Negation(lt)

# ╔═╡ 74fcd796-1d3e-11eb-2412-67bdb49a959a
ρt(not, x; pscale, scale, keepdims, distributed), ρ(not, x; pscale, scale, keepdims, distributed)

# ╔═╡ 74c6595a-1d3e-11eb-2ea0-4397d851d6e9
# And
and = And(subformula1=lt, subformula2=gt)


# ╔═╡ 16d4a0ee-1d3f-11eb-0438-f14a589508a2
ρt(and, (x, y); pscale, scale, keepdims, distributed), ρ(and, (x, y); pscale, scale, keepdims, distributed)

# ╔═╡ 16bdbb22-1d3f-11eb-26fd-49d9bfed3ff9
# Or
or = Or(subformula1=lt, subformula2=gt)


# ╔═╡ 16a4416a-1d3f-11eb-14c3-196fd6889715
ρt(or, (x, y); pscale, scale, keepdims, distributed), ρ(or, (x, y); pscale, scale, keepdims, distributed)

# ╔═╡ 168a71f4-1d3f-11eb-2a2e-6133ad7c7d84
# Always
alw = Always(subformula=gt, interval=nothing)


# ╔═╡ 164ec5bc-1d3f-11eb-1482-d7cd60cefead
ρt(alw, x; pscale, scale, keepdims, distributed), ρ(alw, x; pscale, scale, keepdims, distributed)

# ╔═╡ 5e29f690-1d41-11eb-1ab6-b96b4ffe4171
# Eventually
ev = Always(subformula=gt, interval=nothing)


# ╔═╡ 5e0e0020-1d41-11eb-2c54-95295c0b61a0
ρt(ev, x; pscale, scale, keepdims, distributed), ρ(ev, x; pscale, scale, keepdims, distributed)

# ╔═╡ 2db6a516-1d42-11eb-1378-bf061ecb4496
# Until
ut = Until(subformula1=lt, subformula2=gt)

# ╔═╡ 2d65fec2-1d42-11eb-051b-755f5cbc63eb
ρt(ut, (x,y); pscale, scale, keepdims, distributed), ρ(ut, (x,y); pscale, scale, keepdims, distributed)

# ╔═╡ 5bf3d476-1d42-11eb-25c9-c3ed3ab04a34
# Then
th = Then(subformula1=lt, subformula2=gt)

# ╔═╡ 6323f37a-1d42-11eb-0f71-57dbb9a30e06
ρt(th, (x,y); pscale, scale, keepdims, distributed), ρ(th, (x,y); pscale, scale, keepdims, distributed)

# ╔═╡ 5bd69488-1d42-11eb-0771-0dcfc2655f9d


# ╔═╡ Cell order:
# ╠═d7f2c6b4-1d37-11eb-24aa-2b2499d39686
# ╠═c62ba2e2-1d38-11eb-0ab4-3b41bf022c00
# ╠═b1b1f35c-1d38-11eb-0e94-4f25525b267e
# ╠═88c3412a-1d3e-11eb-2e19-9b70720f9ee6
# ╠═eb614626-1d3c-11eb-351b-a5765ae4513d
# ╠═c3e56212-1d3c-11eb-2d86-a9c88a21ca79
# ╠═e3eaf676-1d3c-11eb-3439-ef7552a13c5e
# ╠═754e9fb6-1d3e-11eb-1e32-4562174f72bb
# ╠═7533e916-1d3e-11eb-2dd1-553a35cbeb07
# ╠═7518b290-1d3e-11eb-22b4-693ab29c73b7
# ╠═74fcd796-1d3e-11eb-2412-67bdb49a959a
# ╠═74c6595a-1d3e-11eb-2ea0-4397d851d6e9
# ╠═16d4a0ee-1d3f-11eb-0438-f14a589508a2
# ╠═16bdbb22-1d3f-11eb-26fd-49d9bfed3ff9
# ╠═16a4416a-1d3f-11eb-14c3-196fd6889715
# ╠═168a71f4-1d3f-11eb-2a2e-6133ad7c7d84
# ╠═164ec5bc-1d3f-11eb-1482-d7cd60cefead
# ╠═5e29f690-1d41-11eb-1ab6-b96b4ffe4171
# ╠═5e0e0020-1d41-11eb-2c54-95295c0b61a0
# ╠═2db6a516-1d42-11eb-1378-bf061ecb4496
# ╠═2d65fec2-1d42-11eb-051b-755f5cbc63eb
# ╠═5bf3d476-1d42-11eb-25c9-c3ed3ab04a34
# ╠═6323f37a-1d42-11eb-0f71-57dbb9a30e06
# ╠═5bd69488-1d42-11eb-0771-0dcfc2655f9d
