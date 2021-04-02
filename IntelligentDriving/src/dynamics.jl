##^# rollout methods ###########################################################
# mutating way
function rollout1(U, x0, f, fx, fu, X_prev, U_prev)
  xdim, N = size(X_prev)
  x = x0
  X = zeros(xdim, N + 1)
  X[:, 1] = x
  for i in 1:N
    x =
      f[:, i] +
      fx[:, :, i] * (x - X_prev[:, i]) +
      fu[:, :, i] * (U[:, i] - U_prev[:, i])
    X[:, i + 1] = x
  end
  return X
end

# tuple way
function rollout2(U, x0, f, fx, fu, X_prev, U_prev)
  xdim, N = size(X_prev)
  x = x0
  xs = (x,)
  for i in 1:N
    x =
      f[:, i] +
      fx[:, :, i] * (x - X_prev[:, i]) +
      fu[:, :, i] * (U[:, i] - U_prev[:, i])
    xs = (xs..., x)
  end
  return reduce(hcat, xs)
end

# cat way
function rollout3(U, x0, f, fx, fu, X_prev, U_prev)
  xdim, N = size(X_prev)
  x = x0
  X = x
  for i in 1:N
    x =
      f[:, i] +
      fx[:, :, i] * (x - X_prev[:, i]) +
      fu[:, :, i] * (U[:, i] - U_prev[:, i])
    X = hcat(X, x)
  end
  return X
end

# Zygote.Buffer way
function rollout4(U, x0, f, fx, fu, X_prev, U_prev)
  xdim, N = size(X_prev)
  x = x0
  x_list = Zygote.Buffer(typeof(x0)[], N + 1)
  x_list[1] = x
  for i in 1:N
    x =
      f[:, i] +
      fx[:, :, i] * (x - X_prev[:, i]) +
      fu[:, :, i] * (U[:, i] - U_prev[:, i])
    x_list[i + 1] = x
  end
  return reduce(hcat, copy(x_list))
end

# select version 4 as the default rollout, it is the best and differentiable
rollout = rollout4
##$#############################################################################
##^# unicycle dynamics #########################################################
function batch_dynamics(
  fn::Function,
  X::AbstractArray{T,2},
  U::AbstractArray{T,2},
  P::AbstractArray{T,2},
) where {T<:Real}
  @assert size(X, 2) == size(U, 2) == size(P, 2)
  return stack(map(i -> fn(X[:, i], U[:, i], P[:, i]), 1:size(X, 2)); dims = -1)
end

@doc raw"""
    unicycle_f(x, u, p) -> f

```math
\begin{aligned}
\dot{x} &= v cos(\theta)
\dot{y} &= v sin(\theta)
\dot{v} &= u_{\text{scale}, 1} u_1
\dot{th} &= u_{\text{scale},1} u_2
\end{aligned}
```

unicycle car dynamics, 4 states:
- `x1`: position x
- `x2`: position y
- `x3`: speed (local frame)
- `x4`: orientation angle
2 actions:
- `u1`: acceleration
- `u2`: turning speed (independent of velocity)
3 parameters:
- `p1`: dt
- `p2`: u1 scaling
- `p3`: u2 scaling

"""
function unicycle_f(
  x::Vector{T},
  u::Vector{T},
  p::Vector{T},
)::Vector{T} where {T<:Real}

  @assert length(x) == 4 && length(u) == 2 && length(p) == 3
  dt, u_scale = p[1], p[2:3]
  u = copy(u)
  eps = 1e-6
  u = u .+ eps * (u .>= 0.0)

  #f = Zygote.Buffer(Float64[], 4)
  f = zeros(T, 4)
  f[1] =
    (
      (u[1] * u_scale[1] * u[2] * u_scale[2] * dt + u[2] * u_scale[2] * x[3]) *
      sin(u[2] * u_scale[2] * dt + x[4]) +
      u[1] * u_scale[1] * cos(u[2] * u_scale[2] * dt + x[4])
    ) / (u[2]^2 * u_scale[2]^2) -
    (u[2] * u_scale[2] * x[3] * sin(x[4]) + u[1] * u_scale[1] * cos(x[4])) /
    (u[2]^2 * u_scale[2]^2) + x[1]
  f[2] =
    (
      u[1] * u_scale[1] * sin(u[2] * u_scale[2] * dt + x[4]) +
      (
        (-u[1] * u_scale[1] * u[2] * u_scale[2] * dt) -
        u[2] * u_scale[2] * x[3]
      ) * cos(u[2] * u_scale[2] * dt + x[4])
    ) / (u[2]^2 * u_scale[2]^2) -
    (u[1] * u_scale[1] * sin(x[4]) - u[2] * u_scale[2] * x[3] * cos(x[4])) /
    (u[2]^2 * u_scale[2]^2) + x[2]
  f[3] = u[1] * u_scale[1] * dt + x[3]
  f[4] = u[2] * u_scale[2] * dt + x[4]
  return copy(f)
end

function unicycle_fx(
  x::Vector{T},
  u::Vector{T},
  p::Vector{T},
)::Matrix{T} where {T<:Real}
  @assert length(x) == 4 && length(u) == 2 && length(p) == 3
  dt, u_scale = p[1], p[2:3]
  u = copy(u)
  eps = 1e-6
  u = u .+ eps * (u .>= 0.0)

  fx = zeros(T, 4, 4)
  fx[1, 1] = 1
  fx[1, 2] = 0
  fx[1, 3] =
    sin(u[2] * u_scale[2] * dt + x[4]) / (u[2] * u_scale[2]) -
    sin(x[4]) / (u[2] * u_scale[2])
  fx[1, 4] =
    (
      (u[1] * u_scale[1] * u[2] * u_scale[2] * dt + u[2] * u_scale[2] * x[3]) *
      cos(u[2] * u_scale[2] * dt + x[4]) -
      u[1] * u_scale[1] * sin(u[2] * u_scale[2] * dt + x[4])
    ) / (u[2]^2 * u_scale[2]^2) -
    (u[2] * u_scale[2] * x[3] * cos(x[4]) - u[1] * u_scale[1] * sin(x[4])) /
    (u[2]^2 * u_scale[2]^2)
  fx[2, 1] = 0
  fx[2, 2] = 1
  fx[2, 3] =
    cos(x[4]) / (u[2] * u_scale[2]) -
    cos(u[2] * u_scale[2] * dt + x[4]) / (u[2] * u_scale[2])
  fx[2, 4] =
    (
      u[1] * u_scale[1] * cos(u[2] * u_scale[2] * dt + x[4]) -
      (
        (-u[1] * u_scale[1] * u[2] * u_scale[2] * dt) -
        u[2] * u_scale[2] * x[3]
      ) * sin(u[2] * u_scale[2] * dt + x[4])
    ) / (u[2]^2 * u_scale[2]^2) -
    (u[2] * u_scale[2] * x[3] * sin(x[4]) + u[1] * u_scale[1] * cos(x[4])) /
    (u[2]^2 * u_scale[2]^2)
  fx[3, 1] = 0
  fx[3, 2] = 0
  fx[3, 3] = 1
  fx[3, 4] = 0
  fx[4, 1] = 0
  fx[4, 2] = 0
  fx[4, 3] = 0
  fx[4, 4] = 1
  return fx
end

function unicycle_fu(
  x::Vector{T},
  u::Vector{T},
  p::Vector{T},
)::Matrix{T} where {T<:Real}
  @assert length(x) == 4 && length(u) == 2 && length(p) == 3
  dt, u_scale = p[1], p[2:3]
  u = copy(u)
  eps = 1e-6
  u = u .+ eps * (u .>= 0.0)

  fu = zeros(T, 4, 2)
  fu[1, 1] =
    (
      u_scale[1] * u[2] * u_scale[2] * dt * sin(u[2] * u_scale[2] * dt + x[4]) +
      u_scale[1] * cos(u[2] * u_scale[2] * dt + x[4])
    ) / (u[2]^2 * u_scale[2]^2) -
    (u_scale[1] * cos(x[4])) / (u[2]^2 * u_scale[2]^2)
  fu[1, 2] =
    (
      -(
        2 * (
          (
            u[1] * u_scale[1] * u[2] * u_scale[2] * dt +
            u[2] * u_scale[2] * x[3]
          ) * sin(u[2] * u_scale[2] * dt + x[4]) +
          u[1] * u_scale[1] * cos(u[2] * u_scale[2] * dt + x[4])
        )
      ) / (u[2]^3 * u_scale[2]^2)
    ) +
    (
      (u[1] * u_scale[1] * u_scale[2] * dt + u_scale[2] * x[3]) *
      sin(u[2] * u_scale[2] * dt + x[4]) -
      u[1] * u_scale[1] * u_scale[2] * dt * sin(u[2] * u_scale[2] * dt + x[4]) +
      u_scale[2] *
      dt *
      (u[1] * u_scale[1] * u[2] * u_scale[2] * dt + u[2] * u_scale[2] * x[3]) *
      cos(u[2] * u_scale[2] * dt + x[4])
    ) / (u[2]^2 * u_scale[2]^2) +
    (
      2 *
      (u[2] * u_scale[2] * x[3] * sin(x[4]) + u[1] * u_scale[1] * cos(x[4]))
    ) / (u[2]^3 * u_scale[2]^2) - (x[3] * sin(x[4])) / (u[2]^2 * u_scale[2])
  fu[2, 1] =
    (
      u_scale[1] * sin(u[2] * u_scale[2] * dt + x[4]) -
      u_scale[1] * u[2] * u_scale[2] * dt * cos(u[2] * u_scale[2] * dt + x[4])
    ) / (u[2]^2 * u_scale[2]^2) -
    (u_scale[1] * sin(x[4])) / (u[2]^2 * u_scale[2]^2)
  fu[2, 2] =
    (
      (
        -u_scale[2] *
        dt *
        (
          (-u[1] * u_scale[1] * u[2] * u_scale[2] * dt) -
          u[2] * u_scale[2] * x[3]
        ) *
        sin(u[2] * u_scale[2] * dt + x[4])
      ) +
      ((-u[1] * u_scale[1] * u_scale[2] * dt) - u_scale[2] * x[3]) *
      cos(u[2] * u_scale[2] * dt + x[4]) +
      u[1] * u_scale[1] * u_scale[2] * dt * cos(u[2] * u_scale[2] * dt + x[4])
    ) / (u[2]^2 * u_scale[2]^2) -
    (
      2 * (
        u[1] * u_scale[1] * sin(u[2] * u_scale[2] * dt + x[4]) +
        (
          (-u[1] * u_scale[1] * u[2] * u_scale[2] * dt) -
          u[2] * u_scale[2] * x[3]
        ) * cos(u[2] * u_scale[2] * dt + x[4])
      )
    ) / (u[2]^3 * u_scale[2]^2) +
    (
      2 *
      (u[1] * u_scale[1] * sin(x[4]) - u[2] * u_scale[2] * x[3] * cos(x[4]))
    ) / (u[2]^3 * u_scale[2]^2) +
    (x[3] * cos(x[4])) / (u[2]^2 * u_scale[2])
  fu[3, 1] = u_scale[1] * dt
  fu[3, 2] = 0
  fu[4, 1] = 0
  fu[4, 2] = u_scale[2] * dt
  return fu
end
##$#############################################################################
##^# dynamics map ##############################################################
DYNAMICS_FN_MAP = Dict{String,Tuple{Function,Function,Function}}(
  "unicycle" => (unicycle_f, unicycle_fx, unicycle_fu),
)
DYNAMICS_DIM_MAP = Dict{String,Tuple{Int,Int}}("unicycle" => (4, 2))
##$#############################################################################
##^# create the SCP dynamics matrix ############################################
function linearized_dynamics(
  x0::AbstractArray{T,1},
  f::AbstractArray{T,2},
  fx::AbstractArray{T,3},
  fu::AbstractArray{T,3},
  X_prev::AbstractArray{T,2},
  U_prev::AbstractArray{T,2},
)::Tuple{Matrix{T},Vector{T}} where {T<:Real}
  xdim, udim, N = size(fu)
  F = zeros(N * xdim, N * udim)
  for i in 1:N
    for j in 1:(i - 1)
      Fm1 = F[
        ((i - 2) * xdim + 1):((i - 1) * xdim),
        ((j - 1) * udim + 1):(j * udim),
      ]
      F[((i - 1) * xdim + 1):(i * xdim), ((j - 1) * udim + 1):(j * udim)] =
        fx[:, :, i] * Fm1
    end
    F[((i - 1) * xdim + 1):(i * xdim), ((i - 1) * udim + 1):(i * udim)] =
      fu[:, :, i]
  end
  f_ = zeros(xdim, N)
  f_[:, 1] = f[:, 1] - fu[:, :, 1] * U_prev[:, 1]
  for i in 2:N
    f_[:, i] =
      f[:, i] + fx[:, :, i] * (f_[:, i - 1] - X_prev[:, i]) -
      fu[:, :, i] * U_prev[:, i]
  end
  return F, reshape(f_, :)
end
##$#############################################################################
