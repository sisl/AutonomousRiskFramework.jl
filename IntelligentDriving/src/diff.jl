##^# utils #####################################################################
function reduce_sum(x; dims = nothing)
  @assert dims != nothing
  return dropdims(sum(x; dims = dims); dims = dims)
end

function stack(x_list; dims = 1)
  @assert length(dims) == 1
  if dims == 1
    return reduce(vcat, [reshape(x, 1, size(x)...) for x in x_list])
  else
    return reduce(
      (a, b) -> cat(a, b; dims = dims),
      [
        reshape(x, size(x)[1:(dims - 1)]..., 1, size(x)[dims:end]...)
        for x in x_list
      ],
    )
  end
end
##$#############################################################################
##^# derivatives ###############################################################
function jacobian_gen(fn; argnums = (), bdims = 0)
  f_fn, f_fn_, fi_fn = nothing, nothing, nothing
  return function (args...)
    (f_fn == nothing) && (f_fn = fn)
    f = f_fn(args...)
    batch_dims = Tuple(ndims(f):-1:(ndims(f) - bdims + 1))
    if bdims > 0
      f = reduce_sum(f; dims = batch_dims)
    end
    @assert typeof(f) <: AbstractArray || typeof(f) <: Number
    if size(f) == () && bdims == 0
      gs = Zygote.gradient(f_fn, args...)
    elseif size(f) == ()
      if f_fn_ == nothing
        f_fn_ = function (args...)
          return reduce_sum(f_fn(args...); dims = batch_dims)
        end
      end
      gs = Zygote.gradient(f_fn_, args...)
    else
      if fi_fn == nothing
        fi_fn = function (i, args...)
          return reshape(reduce_sum(f_fn(args...); dims = batch_dims), :)[i]
        end
      end
      gs_list = [Zygote.gradient(fi_fn, i, args...)[2:end] for i in 1:length(f)]
      gs = [stack(g) for g in zip(gs_list...)]
      gs = [reshape(g, size(f)..., size(g)[2:end]...) for g in gs]
    end
    if !(typeof(argnums) <: Tuple) && size(argnums) == ()
      argnums = [argnums]
    end
    if length(argnums) != 0
      gs = Tuple(g for (i, g) in enumerate(gs) if i in argnums)
    end
    return length(gs) == 0 ? nothing : (length(gs) == 1 ? gs[1] : gs)
  end
end

function hessian_gen(fn; argnums = ())
  return function (args...)
    if length(args) == 1
      hs = Zygote.hessian(fn, args[1])
    else
      @assert argnums == 1
      hs = Zygote.hessian(arg1 -> fn(arg1, args[2:end]...), args[1])
    end
    return hs
  end
end
##$#############################################################################
