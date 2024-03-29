##^# imports ###################################################################
include(joinpath(@__DIR__, "header.jl"))
using STLCG

PSCALE, SCALE = 1e1, 0
##$#############################################################################
##^# utility functions #########################################################
ID.@constdef robustness = STLCG.ρ
ID.@constdef robustness_trace = STLCG.ρt

function eval_robustness_trace(formula, input; settings...)
  #(ndims(input) == 1) && (input = reshape(input, 1, :, 1))
  #return reverse(
  #  robustness_trace(formula, reverse(input; dims = 2); settings...);
  #  dims = 2,
  #)
  return robustness_trace(formula, input; settings...)
end

#function eval_robustness_trace_(formula, input; settings...)
#  return reverse(
#    formula.robustness_trace(
#      torch.as_tensor(reverse(input; dims = 2));
#      settings...,
#    ).detach().cpu().numpy();
#    dims = 2,
#  )
#end
##$#############################################################################
##^# main tests ################################################################
phi = Always(; subformula = LessThan(:r, 0.0), interval = nothing)
#phi_ = stl.Always(stl.LessThan(stl.Expression(:x, nothing), 0.0))
n = 100
t = range(0.0; stop = 10.0, length = n)
sig = reshape(exp.(-t) .- 0.1, 1, :, 1)
display(size(sig))
opts = Dict(:pscale => 1e1, :scale => -1)
rho = eval_robustness_trace(phi, sig; opts...)
#rho_ = eval_robustness_trace_(phi_, sig; opts...)
#println(norm(reshape(rho - rho_, :)))
rho = eval_robustness_trace(
  Always(; subformula = LessThan(:r, 0.0), interval = nothing),
  #LessThan(:r, 0.0),
  reshape(1.0 * ones(n), 1, :, 1);
  #1.0 * ones(n);
  opts...,
)

phi = Always(; subformula = LessThan(:r, 0.0), interval = nothing)

g = Zygote.gradient(
  s -> sum(robustness_trace(phi, reshape(s * ones(n), 1, :, 1);)),
  1.0,
)[1]

closeall()
fig = plot()
plot!(fig, t, reshape(sig, :); label = "sig")
plot!(fig, t, reshape(rho, :); label = "rho")
#plot!(fig, t, reshape(rho_, :); label = "rho_")
display(fig)
##$#############################################################################
