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

# ╔═╡ ec91305e-25f9-11eb-03e3-5f650e18dea1
vcat(STLCG.gradient(x -> sum(ρ(lt, x)), x)...)

# ╔═╡ 754e9fb6-1d3e-11eb-1e32-4562174f72bb
# Greater than
gt = GreaterThan(:x, 0.0)

# ╔═╡ 7533e916-1d3e-11eb-2dd1-553a35cbeb07
ρt(gt, x; pscale, scale, keepdims, distributed), ρ(gt, x; pscale, scale, keepdims, distributed)

# ╔═╡ 727e0406-25fa-11eb-057e-47d41344478d
vcat(STLCG.gradient(x -> sum(ρ(gt, x)), x)...)

# ╔═╡ 7518b290-1d3e-11eb-22b4-693ab29c73b7
# Negation
not = Negation(lt)

# ╔═╡ 74fcd796-1d3e-11eb-2412-67bdb49a959a
ρt(not, x; pscale, scale, keepdims, distributed), ρ(not, x; pscale, scale, keepdims, distributed)

# ╔═╡ 6bcc2b08-25fa-11eb-088c-5f02ca1e3a4d
vcat(STLCG.gradient(x -> sum(ρ(not, x)), x)...)

# ╔═╡ 74c6595a-1d3e-11eb-2ea0-4397d851d6e9
# And
and = And(subformula1=lt, subformula2=gt)


# ╔═╡ 16d4a0ee-1d3f-11eb-0438-f14a589508a2
ρt(and, (x, y); pscale, scale, keepdims, distributed), ρ(and, (x, y); pscale, scale, keepdims, distributed)

# ╔═╡ 7c87efba-25fa-11eb-344d-5950ad39e86d
vcat(STLCG.gradient(x -> sum(ρ(and, x)), (x,y))...)

# ╔═╡ 16bdbb22-1d3f-11eb-26fd-49d9bfed3ff9
# Or
or = Or(subformula1=lt, subformula2=gt)


# ╔═╡ 16a4416a-1d3f-11eb-14c3-196fd6889715
ρt(or, (x, y); pscale, scale, keepdims, distributed), ρ(or, (x, y); pscale, scale, keepdims, distributed)

# ╔═╡ ce1439f4-25fa-11eb-286a-fffb78b1e052
vcat(STLCG.gradient(x -> sum(ρ(or, x)), (x,y))...)

# ╔═╡ 168a71f4-1d3f-11eb-2a2e-6133ad7c7d84
# Always
alw = Always(subformula=gt, interval=nothing)


# ╔═╡ 164ec5bc-1d3f-11eb-1482-d7cd60cefead
ρt(alw, x; pscale, scale, keepdims, distributed), ρ(alw, x; pscale, scale, keepdims, distributed)

# ╔═╡ d601b524-25fa-11eb-3953-f57a814d242e
vcat(STLCG.gradient(x -> sum(ρ(alw, x)), x)...)

# ╔═╡ 5e29f690-1d41-11eb-1ab6-b96b4ffe4171
# Eventually
ev = Eventually(subformula=gt, interval=nothing)


# ╔═╡ 5e0e0020-1d41-11eb-2c54-95295c0b61a0
ρt(ev, x; pscale, scale, keepdims, distributed), ρ(ev, x; pscale, scale, keepdims, distributed)

# ╔═╡ df29e3ec-25fa-11eb-2da2-099ea142ac5e
vcat(STLCG.gradient(x -> sum(ρ(ev, x)), x)...)

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
p = [2,3,4]

# ╔═╡ d754c47e-3c21-11eb-11e8-b349077a74ee
ones(5, size(x)[2:end]...)

# ╔═╡ Cell order:
# ╠═d7f2c6b4-1d37-11eb-24aa-2b2499d39686
# ╠═c62ba2e2-1d38-11eb-0ab4-3b41bf022c00
# ╠═b1b1f35c-1d38-11eb-0e94-4f25525b267e
# ╠═88c3412a-1d3e-11eb-2e19-9b70720f9ee6
# ╠═eb614626-1d3c-11eb-351b-a5765ae4513d
# ╠═c3e56212-1d3c-11eb-2d86-a9c88a21ca79
# ╠═e3eaf676-1d3c-11eb-3439-ef7552a13c5e
# ╠═ec91305e-25f9-11eb-03e3-5f650e18dea1
# ╠═754e9fb6-1d3e-11eb-1e32-4562174f72bb
# ╠═7533e916-1d3e-11eb-2dd1-553a35cbeb07
# ╠═727e0406-25fa-11eb-057e-47d41344478d
# ╠═7518b290-1d3e-11eb-22b4-693ab29c73b7
# ╠═74fcd796-1d3e-11eb-2412-67bdb49a959a
# ╠═6bcc2b08-25fa-11eb-088c-5f02ca1e3a4d
# ╠═74c6595a-1d3e-11eb-2ea0-4397d851d6e9
# ╠═16d4a0ee-1d3f-11eb-0438-f14a589508a2
# ╠═7c87efba-25fa-11eb-344d-5950ad39e86d
# ╠═16bdbb22-1d3f-11eb-26fd-49d9bfed3ff9
# ╠═16a4416a-1d3f-11eb-14c3-196fd6889715
# ╠═ce1439f4-25fa-11eb-286a-fffb78b1e052
# ╠═168a71f4-1d3f-11eb-2a2e-6133ad7c7d84
# ╠═164ec5bc-1d3f-11eb-1482-d7cd60cefead
# ╠═d601b524-25fa-11eb-3953-f57a814d242e
# ╠═5e29f690-1d41-11eb-1ab6-b96b4ffe4171
# ╠═5e0e0020-1d41-11eb-2c54-95295c0b61a0
# ╠═df29e3ec-25fa-11eb-2da2-099ea142ac5e
# ╠═2db6a516-1d42-11eb-1378-bf061ecb4496
# ╠═2d65fec2-1d42-11eb-051b-755f5cbc63eb
# ╠═5bf3d476-1d42-11eb-25c9-c3ed3ab04a34
# ╠═6323f37a-1d42-11eb-0f71-57dbb9a30e06
# ╠═5bd69488-1d42-11eb-0771-0dcfc2655f9d
# ╠═d754c47e-3c21-11eb-11e8-b349077a74ee
