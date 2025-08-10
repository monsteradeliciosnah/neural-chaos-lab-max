import numpy as np

def lorenz(n=5000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = 1.0, 1.0, 1.0
    xs, ys, zs = [], [], []
    for _ in range(n):
        dx = sigma*(y-x)
        dy = x*(rho - z) - y
        dz = x*y - beta*z
        x += dx*dt; y += dy*dt; z += dz*dt
        xs.append(x); ys.append(y); zs.append(z)
    return np.stack([xs,ys,zs], axis=1)

def rossler(n=5000, dt=0.01, a=0.2, b=0.2, c=5.7):
    x, y, z = 1.0, 1.0, 1.0
    xs, ys, zs = [], [], []
    for _ in range(n):
        dx = -y - z
        dy = x + a*y
        dz = b + z*(x - c)
        x += dx*dt; y += dy*dt; z += dz*dt
        xs.append(x); ys.append(y); zs.append(z)
    return np.stack([xs,ys,zs], axis=1)
