### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 83a0bf80-d2d1-11eb-3510-4fea65d71bb9
using TikzNeuralNetworks

# ╔═╡ 535b4e95-d35c-4d86-b5cd-9d79e61694d4
nn = TikzNeuralNetwork(input_size=1,
                       input_label=i->L"s \in \mathcal{S}",
                       hidden_layer_sizes=[2, 4, 3, 4],
                       hidden_layer_labels=(h,i)->["{\\scriptsize\$a_{$j}^{[$h]}\$}" for j in 1:i],
                       output_size=1,
                       output_label=i->L"a \in \mathcal{A}",
                       node_size="24pt")

# ╔═╡ 7c001300-0c04-4c44-b421-91f9c52bafb3
save(PDF("ppo.pdf"), nn)

# ╔═╡ Cell order:
# ╠═83a0bf80-d2d1-11eb-3510-4fea65d71bb9
# ╠═535b4e95-d35c-4d86-b5cd-9d79e61694d4
# ╠═7c001300-0c04-4c44-b421-91f9c52bafb3
