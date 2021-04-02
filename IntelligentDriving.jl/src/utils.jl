##^# Python extensions support #################################################
pushfirst!(
  PyVector(pyimport("sys")."path"),
  joinpath(@__DIR__, "../resources/python"),
)
py_dynamics = pyimport("dynamics")
##$#############################################################################
##^# defining constants without redefinition error #############################
to_sym(var) = Meta.parse(":" * "$var")
function join_dand(exprs...)
  expr1 = :(!isdefined(Main, $(to_sym(exprs[1]))))
  return length(exprs) == 1 ? expr1 : :($expr1 && $(join_dand(exprs[2:end]...)))
end
constdef_assign(exp) = (@assert exp.head == :(=); Expr(:const, exp))
macro constdef(exp)
  var = exp.args[1]
  cond = typeof(var) == Expr ? join_dand(var.args...) : join_dand(var)
  expr = quote
    if $(cond)
      $(constdef_assign(exp))
    end
  end
  return esc(expr)
end
##$#############################################################################
