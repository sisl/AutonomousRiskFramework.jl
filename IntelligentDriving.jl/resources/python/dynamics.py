##^# library imports ###########################################################
import os, pdb, sys, time
import matplotlib.pyplot as plt, numpy as np, torch

##$#############################################################################
##^# dynamics ##################################################################
def car(x, u, p, pkg=torch):
    """
    unicycle car dynamics, 4 states, 2 actions
    x1: position x
    x2: position y
    x3: speed (local frame)
    x4: orientation angle

    u1: acceleration
    u2: turning speed (independent of velocity)
    """
    assert x.shape[-1] == 4 and u.shape[-1] == 2
    T, u_scale1, u_scale2 = p[..., 0], p[..., 1], p[..., 2]
    eps = 1e-6 * pkg.ones(())
    u1, u2 = u_scale1 * u[..., 0], -u_scale2 * u[..., 1]
    u1 = u1 + pkg.where(u1 >= 0.0, eps, -eps)
    u2 = u2 + pkg.where(u2 >= 0.0, eps, -eps)

    x0, y0, v0, th0 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xp = pkg.zeros(x.shape)
    xp1 = (
        x0
        + (
            u1 * pkg.sin(T * u1 + th0) * v0
            + (T * u1 * pkg.sin(T * u1 + th0) + pkg.cos(T * u1 + th0)) * u2
        )
        / u1 ** 2
        - (pkg.sin(th0) * u1 * v0 + pkg.cos(th0) * u2) / u1 ** 2
    )
    xp2 = (
        y0
        + (pkg.cos(th0) * u1 * v0 - pkg.sin(th0) * u2) / u1 ** 2
        - (
            u1 * pkg.cos(T * u1 + th0) * v0
            + (T * u1 * pkg.cos(T * u1 + th0) - pkg.sin(T * u1 + th0)) * u2
        )
        / u1 ** 2
    )
    xp3 = v0 + T * u1
    xp4 = th0 + T * u2
    xp = pkg.cat(
        [xp1[..., None], xp2[..., None], xp3[..., None], xp4[..., None]], -1
    )
    return xp

def hello():
    print("Hello World")


def f_fx_fu_fn(x, u, p):
    """
    This function accomodates Julia style batched inputs (xdim,) + bshape
    """
    xdim, udim, bshape = x.shape[0], u.shape[0], x.shape[1:]
    assert bshape == u.shape[1:] and bshape == p.shape[1:]

    x, u, p = [z.reshape((z.shape[0], -1)).swapaxes(-2, -1) for z in [x, u, p]]
    x, u, p = torch.as_tensor(x), torch.as_tensor(u), torch.as_tensor(p)
    x.requires_grad = True
    u.requires_grad = True
    f = car(x, u, p)
    gs_list = zip(
        *[
            torch.autograd.grad(
                torch.sum(f, tuple(range(f.ndim - 1)))[i],
                (x, u),
                retain_graph=(i - 1 < f.shape[-1]),
            )
            for i in range(f.shape[-1])
        ]
    )
    fx, fu = [torch.stack(gs, -2) for gs in gs_list]
    fx = (
        fx.reshape((-1, xdim * xdim))
        .transpose(-2, -1)
        .reshape((xdim, xdim) + bshape)
    )
    fu = (
        fu.reshape((-1, xdim * udim))
        .transpose(-2, -1)
        .reshape((xdim, udim) + bshape)
    )
    f = f.detach().transpose(-2, -1).reshape((xdim,) + bshape)
    return f.numpy(), fx.numpy(), fu.numpy()


##$#############################################################################
##^# linearization tools #######################################################
def dyn_mat(x0, f, fx, fu, X_prev, U_prev, pkg=torch):
    """
    construct the matrix and bias vector that gives from a local linearization
    vec(X) = Ft @ vec(U) + ft
    """
    bshape, (N, xdim), udim = fx.shape[:-3], fx.shape[-3:-1], fu.shape[-1]
    Fts = [[None for _ in range(N)] for _ in range(N)]
    Z_ = pkg.zeros(bshape + (xdim, udim))
    Fts = [[Z_ for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(i):
            Fts[i][j] = fx[..., i, :, :] @ Fts[i - 1][j]
        Fts[i][i] = fu[..., i, :, :]
    Ft = pkg.cat(
        [pkg.cat([Fts[i][j] for i in range(N)], -2) for j in range(N)], -1
    )

    fts = [None for i in range(N)]
    f_ = f - bmv(fx, X_prev) - bmv(fu, U_prev)
    fts[0] = bmv(fx[..., 0, :, :], x0) + f_[..., 0, :]
    for i in range(1, N):
        fts[i] = bmv(fx[..., i, :, :], fts[i - 1]) + f_[..., i, :]
    ft = pkg.cat(fts, -1)
    return Ft, ft


##$#############################################################################
##^# tests #####################################################################
if __name__ == "__main__":
    bshape = (60,)
    X = np.random.randn(*((4,) + bshape))
    U = np.random.randn(*((2,) + bshape))
    p = 0.1 * np.ones((1,) + bshape)
    t = time.time()
    M = 10 ** 2
    for _ in range(M):
        f, fx, fu = f_fx_fu_fn(X, U, p)
    t = time.time() - t
    print("Time elapsed: %9.4e" % (t / M))
    print(f.shape)
    print(fx.shape)
    print(fu.shape)
    print()
    print()
    print()
    print(fx[:, :, 0])
    print(fu[:, :, 0])
##$#############################################################################
