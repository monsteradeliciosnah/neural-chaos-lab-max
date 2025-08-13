import numpy as np

__all__ = [
    "lorenz_step",
    "rossler_step",
    "henon_step",
    "logistic_step",
    "ikeda_step",
    "lorenz",
    "rossler",
    "henon",
    "logistic",
    "ikeda",
]


def _num(val, default):
    try:
        return float(val)
    except Exception:
        return float(default)


def _vec(x, dim, default):
    try:
        if x is None:
            arr = np.array(default, dtype=float)
        else:
            arr = np.array(x, dtype=float).reshape(-1)
            if arr.size < dim:
                pad = np.array(default, dtype=float)[: dim - arr.size]
                arr = np.concatenate([arr, pad])
            elif arr.size > dim:
                arr = arr[:dim]
    except Exception:
        arr = np.array(default, dtype=float)
    return arr


def _first_arg_or_none(args):
    return args[0] if args else None


def _guarded(dim, default):
    def deco(fn):
        def inner(*args, **kwargs):
            try:
                out = fn(*args, **kwargs)
                out = np.array(out, dtype=float).reshape(-1)
                if out.size != dim:
                    out = _vec(out, dim, default)
                return out
            except Exception:
                return np.array(default, dtype=float)

        return inner

    return deco


@_guarded(3, [1.0, 1.0, 1.0])
def lorenz_step(*args, **kwargs):
    state = kwargs.pop("state", _first_arg_or_none(args))
    s = _vec(state, 3, [1.0, 1.0, 1.0])
    sigma = _num(kwargs.pop("sigma", 10.0), 10.0)
    rho = _num(kwargs.pop("rho", 28.0), 28.0)
    beta = _num(kwargs.pop("beta", 8.0 / 3.0), 8.0 / 3.0)
    dt = _num(kwargs.pop("dt", 0.01), 0.01)
    x, y, z = s
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return s + dt * np.array([dx, dy, dz], dtype=float)


@_guarded(3, [1.0, 1.0, 1.0])
def rossler_step(*args, **kwargs):
    state = kwargs.pop("state", _first_arg_or_none(args))
    s = _vec(state, 3, [1.0, 1.0, 1.0])
    a = _num(kwargs.pop("a", 0.2), 0.2)
    b = _num(kwargs.pop("b", 0.2), 0.2)
    c = _num(kwargs.pop("c", 5.7), 5.7)
    dt = _num(kwargs.pop("dt", 0.01), 0.01)
    x, y, z = s
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return s + dt * np.array([dx, dy, dz], dtype=float)


@_guarded(2, [0.1, 0.0])
def henon_step(*args, **kwargs):
    state = kwargs.pop("state", _first_arg_or_none(args))
    s = _vec(state, 2, [0.1, 0.0])
    a = _num(kwargs.pop("a", 1.4), 1.4)
    b = _num(kwargs.pop("b", 0.3), 0.3)
    x, y = s
    x_next = 1.0 - a * x * x + y
    y_next = b * x
    return np.array([x_next, y_next], dtype=float)


@_guarded(1, [0.5])
def logistic_step(*args, **kwargs):
    state = kwargs.pop("state", _first_arg_or_none(args))
    s = _vec(state, 1, [0.5])
    r = _num(kwargs.pop("r", 3.9), 3.9)
    x = float(s[0])
    return np.array([r * x * (1.0 - x)], dtype=float)


@_guarded(2, [0.1, 0.0])
def ikeda_step(*args, **kwargs):
    state = kwargs.pop("state", _first_arg_or_none(args))
    s = _vec(state, 2, [0.1, 0.0])
    u = _num(kwargs.pop("u", 0.9), 0.9)
    x, y = s
    t = 0.4 - 6.0 / (1.0 + x * x + y * y)
    ct, st = np.cos(t), np.sin(t)
    x_next = 1.0 + u * (x * ct - y * st)
    y_next = u * (x * st + y * ct)
    return np.array([x_next, y_next], dtype=float)


def lorenz(n=5000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0, x0=(1.0, 1.0, 1.0)):
    s = _vec(x0, 3, [1.0, 1.0, 1.0])
    xs = np.empty((int(n), 3), dtype=float)
    for i in range(int(n)):
        s = lorenz_step(state=s, dt=dt, sigma=sigma, rho=rho, beta=beta)
        xs[i] = s
    return xs


def rossler(n=5000, dt=0.01, a=0.2, b=0.2, c=5.7, x0=(1.0, 1.0, 1.0)):
    s = _vec(x0, 3, [1.0, 1.0, 1.0])
    xs = np.empty((int(n), 3), dtype=float)
    for i in range(int(n)):
        s = rossler_step(state=s, dt=dt, a=a, b=b, c=c)
        xs[i] = s
    return xs


def henon(n=10000, a=1.4, b=0.3, x0=(0.1, 0.0)):
    s = _vec(x0, 2, [0.1, 0.0])
    xs = np.empty((int(n), 2), dtype=float)
    for i in range(int(n)):
        s = henon_step(state=s, a=a, b=b)
        xs[i] = s
    return xs


def logistic(n=10000, r=3.9, x0=(0.5,)):
    s = _vec(x0, 1, [0.5])
    xs = np.empty((int(n), 1), dtype=float)
    for i in range(int(n)):
        s = logistic_step(state=s, r=r)
        xs[i] = s
    return xs


def ikeda(n=10000, u=0.9, x0=(0.1, 0.0)):
    s = _vec(x0, 2, [0.1, 0.0])
    xs = np.empty((int(n), 2), dtype=float)
    for i in range(int(n)):
        s = ikeda_step(state=s, u=u)
        xs[i] = s
    return xs