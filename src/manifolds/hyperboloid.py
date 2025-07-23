import torch

EPS = 1e-15


# x,y: (..., d + 1)
def minkowski_dot(x, y):
    return -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)


def lorentz_norm(v):
    return torch.sqrt(torch.clamp(minkowski_dot(v, v), min=EPS))


# Prop 3.2
# tangent space => hyperbolic
def expmap(x, v, K):
    sqrtK = torch.sqrt(torch.as_tensor(K, dtype=x.dtype, device=x.device))
    v_norm = lorentz_norm(v)
    coef1 = torch.cosh(v_norm / sqrtK)[..., None]
    coef2 = torch.sinh(v_norm / sqrtK)[..., None] * sqrtK / (v_norm[..., None] + EPS)
    return coef1 * x + coef2 * v


# hyperbolic => tangent
def logmap(x, y, K):
    sqrtK = torch.sqrt(torch.as_tensor(K, dtype=x.dtype, device=x.device))
    xy = minkowski_dot(x, y)
    dist = sqrtK * torch.acosh(torch.clamp(-xy / K, min=1.0 + 1e-7))  # d^K(x,y)
    u = y + (xy / K)[..., None] * x
    u_norm = lorentz_norm(u)
    return (dist / (u_norm + EPS))[..., None] * u


# origin-based versions (x = north pole o = (sqrtK, 0,...,0)):
def expmap0(u, K):
    sqrtK = torch.sqrt(torch.as_tensor(K, dtype=u.dtype, device=u.device))
    u_norm = lorentz_norm(u)
    coef0 = torch.cosh(u_norm / sqrtK)
    coef1 = torch.sinh(u_norm / sqrtK) * sqrtK / (u_norm + EPS)
    return torch.cat([coef0[..., None] * sqrtK, coef1[..., None] * u[..., 1:]], dim=-1)


def logmap0(y, K):
    sqrtK = torch.sqrt(torch.as_tensor(K, dtype=y.dtype, device=y.device))
    o = torch.zeros_like(y)
    o[..., 0] = sqrtK
    return logmap(o, y, K)
