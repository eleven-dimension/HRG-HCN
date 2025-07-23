import pytest
import torch
from torch.testing import assert_close

from manifolds import expmap, logmap, expmap0, logmap0, minkowski_dot, lorentz_norm

MANIFOLD_ATOL = 1e-6
MANIFOLD_RTOL = 1e-6
ROUNDTRIP_ATOL = 1e-6
ROUNDTRIP_RTOL = 1e-6
RESIDUAL_MAX = 5e-3
TEST_EPS = 1e-10


def rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype):
    z = torch.randn(batch_shape + (dim,), device=device, dtype=dtype)
    t = torch.sqrt(K + (z**2).sum(-1, keepdim=True))
    return torch.cat([t, z], dim=-1)


def origin_point(K, dim, device, dtype):
    o = torch.zeros(dim + 1, device=device, dtype=dtype)
    o[0] = torch.sqrt(K)
    return o


def project_to_tangent(x, w, K):
    alpha = minkowski_dot(x, w) / (-K)
    return w - alpha.unsqueeze(-1) * x


def lorentz_norm_safe(v):
    return torch.sqrt(torch.clamp(minkowski_dot(v, v), min=TEST_EPS))


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture(params=[1.0, 0.5, 5.0])
def K(request, device):
    return torch.tensor(request.param, dtype=torch.double, device=device)


@pytest.fixture(params=[2, 5, 20])
def dim(request):
    return request.param


@pytest.fixture(params=[(), (7,), (3, 2)])
def batch_shape(request):
    return request.param


def test_expmap_stays_on_manifold(device, K, dim, batch_shape):
    torch.manual_seed(0)
    dtype = torch.double
    x = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)
    w = torch.randn_like(x)

    w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v = project_to_tangent(x, w, K)
    y = expmap(x, v, K)

    lhs = minkowski_dot(y, y)
    residual = torch.abs(lhs + K)
    assert (
        residual.max() < RESIDUAL_MAX
    ), f"Manifold residual too large: {residual.max().item()}"
    assert_close(lhs, -K.expand_as(lhs), atol=RESIDUAL_MAX, rtol=1e-5)


def test_logmap_returns_tangent(device, K, dim, batch_shape):
    torch.manual_seed(1)
    dtype = torch.double
    x = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)
    y = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)
    v = logmap(x, y, K)
    dot = minkowski_dot(x, v)
    assert_close(dot, torch.zeros_like(dot), atol=MANIFOLD_ATOL, rtol=MANIFOLD_RTOL)


def test_round_trip_xy(device, K, dim, batch_shape):
    torch.manual_seed(2)
    dtype = torch.double
    x = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)
    y = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)

    v = logmap(x, y, K)
    y_rec = expmap(x, v, K)
    assert_close(y_rec, y, atol=ROUNDTRIP_ATOL, rtol=ROUNDTRIP_RTOL)


def test_round_trip_xv(device, K, dim, batch_shape):
    torch.manual_seed(3)
    dtype = torch.double
    x = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)
    w = torch.randn_like(x)
    v = project_to_tangent(x, w, K)

    y = expmap(x, v, K)
    v_rec = logmap(x, y, K)
    assert_close(v_rec, v, atol=ROUNDTRIP_ATOL, rtol=ROUNDTRIP_RTOL)


def test_small_norm_limit(device, K, dim, batch_shape):
    torch.manual_seed(4)
    dtype = torch.double
    x = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)
    w = torch.randn_like(x)
    v = project_to_tangent(x, w, K) * 1e-8

    y = expmap(x, v, K)
    max_dev = (y - x).abs().max().item()
    assert max_dev < 1e-5, f"expmap small-norm limit deviates by {max_dev}"

    v0 = logmap(x, x, K)
    assert torch.max(torch.abs(v0)) < 1e-8


def test_broadcasting(device, K, dim):
    torch.manual_seed(5)
    dtype = torch.double
    x = rand_point_on_hyperboloid((4,), dim, K, device, dtype)
    w = torch.randn(1, 4, dim + 1, device=device, dtype=dtype)
    v = project_to_tangent(x.unsqueeze(0), w, K)
    y = expmap(x.unsqueeze(0), v, K)
    assert y.shape == (1, 4, dim + 1)


def test_distance_consistency(device, K, dim, batch_shape):
    torch.manual_seed(6)
    dtype = torch.double
    x = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)
    y = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)

    xy = minkowski_dot(x, y)
    dist_formula = torch.sqrt(K) * torch.acosh(torch.clamp(-xy / K, min=1.0 + 1e-9))
    v = logmap(x, y, K)
    dist_log = lorentz_norm(v)

    assert_close(dist_formula, dist_log, atol=1e-6, rtol=1e-6)


def test_expmap0_logmap0_roundtrip(device, K, dim, batch_shape):
    torch.manual_seed(9)
    dtype = torch.double
    o = origin_point(K, dim, device, dtype)

    y = rand_point_on_hyperboloid(batch_shape, dim, K, device, dtype)
    u = logmap0(y, K)
    dot = minkowski_dot(o.expand_as(u), u)
    assert_close(dot, torch.zeros_like(dot), atol=MANIFOLD_ATOL, rtol=MANIFOLD_RTOL)

    y_rec = expmap0(u, K)
    assert_close(y_rec, y, atol=ROUNDTRIP_ATOL, rtol=ROUNDTRIP_RTOL)


def test_expmap0_stays_on_manifold(device, K, dim, batch_shape):
    torch.manual_seed(10)
    dtype = torch.double
    o = origin_point(K, dim, device, dtype)
    u = torch.randn(batch_shape + (dim + 1,), device=device, dtype=dtype)
    u = project_to_tangent(o.expand_as(u), u, K)

    y = expmap0(u, K)
    lhs = minkowski_dot(y, y)
    residual = torch.abs(lhs + K)
    assert residual.max() < RESIDUAL_MAX
    assert_close(lhs, -K.expand_as(lhs), atol=RESIDUAL_MAX, rtol=1e-5)


def test_logmap0_zero_at_origin(device, K, dim):
    torch.manual_seed(11)
    dtype = torch.double
    o = origin_point(K, dim, device, dtype)
    u0 = logmap0(o, K)
    assert torch.max(torch.abs(u0)) < 1e-10


@pytest.mark.slow
def test_gradcheck_expmap(device, K):
    torch.manual_seed(12)
    dtype = torch.double
    dim = 3
    x = rand_point_on_hyperboloid((), dim, K, device, dtype).requires_grad_(True)
    w = torch.randn_like(x)
    v = project_to_tangent(x.detach(), w, K).requires_grad_(True)

    def func(xx, vv):
        return expmap(xx, vv, K)

    assert torch.autograd.gradcheck(func, (x, v), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.slow
def test_gradcheck_logmap(device, K):
    torch.manual_seed(13)
    dtype = torch.double
    dim = 3
    x = rand_point_on_hyperboloid((), dim, K, device, dtype).requires_grad_(True)
    y = rand_point_on_hyperboloid((), dim, K, device, dtype).requires_grad_(True)

    def func(xx, yy):
        return logmap(xx, yy, K)

    assert torch.autograd.gradcheck(func, (x, y), eps=1e-6, atol=1e-4, rtol=1e-3)
