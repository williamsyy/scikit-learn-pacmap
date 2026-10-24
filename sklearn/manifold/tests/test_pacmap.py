import warnings

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from sklearn import config_context
from sklearn.datasets import load_digits, make_blobs
from sklearn.manifold import PaCMAP, trustworthiness
from sklearn.manifold._pacmap import _phase_weights
from sklearn.manifold._pacmap_utils import (
    _build_incidence,
    _pacmap_gradient,
    _sample_fp_pairs,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.utils._testing import assert_allclose, assert_array_equal

FAST = dict(num_iters=(20, 20, 40))


def _incidence(pairs, types, n):
    return _build_incidence(
        np.ascontiguousarray(pairs, dtype=np.intp),
        np.ascontiguousarray(types, dtype=np.int8),
        n,
    )


# Reproducibility


def test_same_seed_bitwise_identical(global_random_seed):
    X, _ = make_blobs(300, n_features=20, centers=5, random_state=0)
    a = PaCMAP(random_state=global_random_seed, **FAST).fit_transform(X)
    b = PaCMAP(random_state=global_random_seed, **FAST).fit_transform(X)
    assert_array_equal(a, b)


def test_different_seeds_differ():
    X, _ = make_blobs(300, n_features=20, centers=5, random_state=0)
    a = PaCMAP(random_state=0, init="random", **FAST).fit_transform(X)
    b = PaCMAP(random_state=1, init="random", **FAST).fit_transform(X)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("metric", ["euclidean", "angular"])
def test_output_independent_of_thread_count(metric):
    threadpoolctl = pytest.importorskip("threadpoolctl")
    # Integer data has many tied distances.
    rng = np.random.RandomState(0)
    X = np.r_[
        make_blobs(300, n_features=150, centers=4, random_state=0)[0],
        rng.randint(0, 3, size=(200, 150)).astype(float),
    ]
    with threadpoolctl.threadpool_limits(limits=1, user_api="openmp"):
        a = PaCMAP(metric=metric, random_state=0, **FAST).fit_transform(X)
    with threadpoolctl.threadpool_limits(limits=4, user_api="openmp"):
        b = PaCMAP(metric=metric, random_state=0, **FAST).fit_transform(X)
    assert_array_equal(a, b)


def test_random_state_instance_is_supported():
    X = np.random.RandomState(0).rand(100, 4)
    a = PaCMAP(random_state=np.random.RandomState(3), **FAST).fit_transform(X)
    b = PaCMAP(random_state=np.random.RandomState(3), **FAST).fit_transform(X)
    assert_array_equal(a, b)


# Pair construction


def test_pair_invariants(global_random_seed):
    X, _ = make_blobs(400, n_features=8, random_state=global_random_seed)
    est = PaCMAP(random_state=global_random_seed, num_iters=(0, 0, 0)).fit(X)
    n = X.shape[0]
    nn = est.pair_neighbors_[:, 1].reshape(n, est.n_neighbors_)
    mn = est.pair_mn_[:, 1].reshape(n, est.n_mn_)
    fp = est.pair_fp_[:, 1].reshape(n, est.n_fp_)
    rows = np.arange(n)[:, None]
    for block in (nn, mn, fp):
        assert not np.any(block == rows), "self-pair"
        assert all(len(set(r)) == len(r) for r in block), "duplicate partner"
    for i in range(n):
        assert not set(fp[i]) & set(nn[i]), "FP pair overlaps NN pair"


def test_pair_distance_ordering():
    # MN partners are the 2nd closest of 6 random candidates: farther than NN
    # partners and closer than uniform FP partners on average.
    X, _ = make_blobs(1000, n_features=5, centers=10, random_state=0)
    est = PaCMAP(random_state=0, num_iters=(0, 0, 0)).fit(X)

    def mean_dist(P):
        return np.linalg.norm(X[P[:, 0]] - X[P[:, 1]], axis=1).mean()

    assert (
        mean_dist(est.pair_neighbors_)
        < mean_dist(est.pair_mn_)
        < mean_dist(est.pair_fp_)
    )


def test_precomputed_matches_euclidean():
    # Min-max scaling and centering leave the scaled re-ranking and the MN
    # ranking unchanged, so both code paths sample identical pairs.
    X = np.random.RandomState(0).rand(250, 6)
    kw = dict(init="random", random_state=0, **FAST)
    a = PaCMAP(**kw).fit(X)
    b = PaCMAP(metric="precomputed", **kw).fit(cdist(X, X))
    assert_array_equal(a.pair_neighbors_, b.pair_neighbors_)
    assert_array_equal(a.pair_mn_, b.pair_mn_)
    assert_array_equal(a.pair_fp_, b.pair_fp_)
    assert_allclose(a.embedding_, b.embedding_, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "n_samples, expected", [(5000, 10), (10000, 10), (100000, 25), (1000000, 40)]
)
def test_auto_n_neighbors(n_samples, expected):
    assert PaCMAP(n_neighbors=None)._n_pairs(n_samples)[0] == expected


def test_sample_fp_pairs_excludes_neighbors():
    n = 50
    nbrs = np.ascontiguousarray(
        (np.arange(n)[:, None] + np.arange(1, 11)) % n, dtype=np.intp
    )
    fp = _sample_fp_pairs(nbrs, 20, 0, 1)[:, 1].reshape(n, 20)
    for i in range(n):
        assert i not in fp[i]
        assert not set(fp[i]) & set(nbrs[i])
        assert len(set(fp[i])) == 20


def test_build_incidence_matches_stable_argsort():
    rng = np.random.RandomState(0)
    n = 200
    pairs = rng.randint(0, n, size=(3000, 2))
    types = rng.randint(0, 3, 3000)
    indptr, other, t = _incidence(pairs, types, n)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    order = np.argsort(rows, kind="stable")
    assert_array_equal(indptr, np.r_[0, np.cumsum(np.bincount(rows, minlength=n))])
    assert_array_equal(other, np.concatenate([pairs[:, 1], pairs[:, 0]])[order])
    assert_array_equal(t, np.concatenate([types, types])[order])


# Optimization


def test_phase_schedule():
    assert _phase_weights(0, 100, 100) == (2.0, 1000.0, 1.0)
    assert _phase_weights(50, 100, 100) == pytest.approx((2.0, 501.5, 1.0))
    assert _phase_weights(150, 100, 100) == (3.0, 3.0, 1.0)
    assert _phase_weights(200, 100, 100) == (1.0, 0.0, 1.0)


def test_gradient_matches_finite_differences(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n = 30
    Y = rng.randn(n, 2) * 2
    pairs = np.column_stack([np.repeat(np.arange(n), 6), rng.randint(0, n, 6 * n)])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    indptr, other, t = _incidence(pairs, rng.randint(0, 3, len(pairs)), n)
    w = (2.0, 7.0, 1.5)

    def loss_and_grad(Y):
        g, node_loss = np.empty_like(Y), np.empty(n)
        _pacmap_gradient(np.ascontiguousarray(Y), indptr, other, t, *w, g, node_loss, 1)
        return 0.5 * node_loss.sum(), g

    _, grad = loss_and_grad(Y)
    eps = 1e-6
    num = np.empty_like(Y)
    for i in range(n):
        for k in range(2):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[i, k] += eps
            Ym[i, k] -= eps
            num[i, k] = (loss_and_grad(Yp)[0] - loss_and_grad(Ym)[0]) / (2 * eps)
    assert_allclose(grad, num, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("num_iters", [(0, 0, 0), (0, 0, 1), (1, 0, 0), (3, 7, 13)])
def test_loss_is_final_and_finite(num_iters):
    X = np.random.RandomState(0).rand(100, 5)
    est = PaCMAP(num_iters=num_iters, random_state=0).fit(X)
    assert np.isfinite(est.loss_)
    assert est.n_iter_ == sum(num_iters)


def test_num_iters_int():
    X = np.random.RandomState(0).rand(50, 3)
    assert PaCMAP(num_iters=7, random_state=0).fit(X).n_iter_ == 207


def test_divergence_raises(monkeypatch):
    from sklearn.manifold import _pacmap

    def explode(self, Y, grad, itr):
        Y[:] = np.inf

    monkeypatch.setattr(_pacmap._Adam, "update", explode)
    with pytest.raises(ValueError, match="diverged"):
        PaCMAP(**FAST).fit(np.random.RandomState(0).rand(50, 3))


# Embedding quality


def test_digits_quality():
    X, y = load_digits(return_X_y=True)
    emb = PaCMAP(random_state=0).fit_transform(X)
    assert trustworthiness(X, emb, n_neighbors=10) > 0.97
    assert cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean() > 0.95


@pytest.mark.parametrize("metric", ["manhattan", "angular"])
def test_other_metrics_quality(metric):
    X, y = load_digits(return_X_y=True)
    emb = PaCMAP(metric=metric, random_state=0).fit_transform(X)
    assert cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean() > 0.9


def test_hamming_metric():
    rng = np.random.RandomState(0)
    centers = rng.rand(4, 64) < 0.5
    y = rng.randint(0, 4, 400)
    X = np.logical_xor(centers[y], rng.rand(400, 64) < 0.1).astype(float)
    emb = PaCMAP(metric="hamming", random_state=0, **FAST).fit_transform(X)
    assert cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean() > 0.95


def test_pca_preprocessing_path():
    X, y = make_blobs(600, n_features=300, centers=5, random_state=0)
    emb = PaCMAP(random_state=0, **FAST).fit_transform(X)
    assert cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean() > 0.99


@pytest.mark.parametrize("n_components", [1, 3, 10])
def test_n_components(n_components):
    X = np.random.RandomState(0).rand(150, 30)
    Y = PaCMAP(n_components=n_components, random_state=0, **FAST).fit_transform(X)
    assert Y.shape == (150, n_components)
    assert np.all(np.isfinite(Y))


# Corner cases (several mirror past TSNE issues)


@pytest.mark.parametrize("X", [np.ones((10, 5)), np.zeros((91, 3)), np.ones((10, 200))])
def test_identical_samples_warn_and_stay_finite(X):
    # TSNE produced NaN and crashed on constant data (#28368, #32788).
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.warns(UserWarning, match="All samples are identical"):
            Y = PaCMAP(random_state=0, **FAST).fit_transform(X)
    assert np.all(np.isfinite(Y))


def test_heavy_duplicates():
    # >= 6 copies of each point make the local scale sigma_i zero.
    X = np.repeat(np.random.RandomState(0).rand(20, 5), 30, axis=0)
    Y = PaCMAP(random_state=0, **FAST).fit_transform(X)
    assert np.all(np.isfinite(Y))
    assert np.ptp(Y, axis=0).min() > 1


@pytest.mark.parametrize("n_samples", [3, 4, 5, 8, 12, 30])
def test_tiny_datasets(n_samples):
    X = np.random.RandomState(0).rand(n_samples, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Y = PaCMAP(random_state=0, **FAST).fit_transform(X)
    assert Y.shape == (n_samples, 2)
    assert np.all(np.isfinite(Y))


def test_small_n_warns_about_reduced_pairs():
    with pytest.warns(UserWarning, match="too small for the requested pairs"):
        PaCMAP(random_state=0, **FAST).fit(np.random.RandomState(0).rand(12, 3))


def test_no_warning_in_regular_use():
    X = np.random.RandomState(0).rand(200, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PaCMAP(random_state=0, **FAST).fit(X)


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_extreme_scales(scale):
    X = np.random.RandomState(0).rand(200, 5)
    ref = PaCMAP(random_state=0, **FAST).fit_transform(X)
    assert_allclose(
        PaCMAP(random_state=0, **FAST).fit_transform(X * scale), ref, atol=1e-6
    )


def test_float32_and_float64_input_agree():
    X = np.random.RandomState(0).rand(200, 5).astype(np.float32)
    a = PaCMAP(random_state=0, **FAST).fit_transform(X)
    b = PaCMAP(random_state=0, **FAST).fit_transform(X.astype(np.float64))
    assert a.dtype == b.dtype == np.float64
    assert_array_equal(a, b)


@pytest.mark.parametrize("n_features", [5, 150])
def test_pandas_transform_output(n_features):
    # The internal PCA must not follow set_output (TSNE #25365).
    pd = pytest.importorskip("pandas")
    X = np.random.RandomState(0).rand(200, n_features)
    expected = PaCMAP(random_state=0, **FAST).fit_transform(X)
    with config_context(transform_output="pandas"):
        Y = PaCMAP(random_state=0, **FAST).fit_transform(X)
    assert isinstance(Y, pd.DataFrame)
    assert list(Y.columns) == ["pacmap0", "pacmap1"]
    assert_array_equal(Y.to_numpy(), expected)


def test_precomputed_huge_distances_no_overflow():
    P = np.random.RandomState(0).rand(100, 3)
    D = cdist(P, P)
    kw = dict(metric="precomputed", init="random", random_state=0, **FAST)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        a = PaCMAP(**kw).fit_transform(D)
        b = PaCMAP(**kw).fit_transform(D * 1e200)
    assert_array_equal(a, b)


def test_array_init():
    X = np.random.RandomState(0).rand(50, 3)
    init = np.random.RandomState(1).rand(50, 2)
    assert PaCMAP(init=init, random_state=0, **FAST).fit_transform(X).shape == (50, 2)


def test_constant_init_column_warns():
    X = np.random.RandomState(0).rand(50, 4)
    init = np.c_[np.random.RandomState(1).rand(50), np.ones(50)]
    with pytest.warns(UserWarning, match="constant column"):
        PaCMAP(init=init, random_state=0, **FAST).fit(X)


def test_candidates_are_exact_neighbors():
    # The NN pairs are drawn from the exact 60 nearest neighbors.
    X = np.random.RandomState(0).rand(300, 4)
    est = PaCMAP(random_state=0, num_iters=(0, 0, 0)).fit(X)
    Xs = (X - X.min()) / (X - X.min()).max()
    cand = NearestNeighbors(n_neighbors=60).fit(Xs).kneighbors(return_distance=False)
    nn = est.pair_neighbors_[:, 1].reshape(300, est.n_neighbors_)
    assert all(set(nn[i]) <= set(cand[i]) for i in range(300))


@pytest.mark.parametrize(
    "params, X, err, match",
    [
        (dict(num_iters=(1, 2)), np.ones((20, 3)), ValueError, "num_iters"),
        (dict(num_iters=(1, 2, -1)), np.ones((20, 3)), ValueError, "num_iters"),
        (
            dict(init=np.ones((5, 2))),
            np.random.RandomState(0).rand(20, 3),
            ValueError,
            "init must have shape",
        ),
        (
            dict(init=np.full((20, 2), np.nan)),
            np.random.RandomState(0).rand(20, 3),
            ValueError,
            "init",
        ),
        (
            dict(metric="precomputed"),
            np.random.RandomState(0).rand(20, 20),
            ValueError,
            'init="pca"',
        ),
        (
            dict(metric="precomputed", init="random"),
            np.random.RandomState(0).rand(20, 5),
            ValueError,
            "square",
        ),
        (
            dict(metric="precomputed", init="random"),
            -np.ones((20, 20)),
            ValueError,
            "Negative",
        ),
        (
            dict(fp_ratio=0.01),
            np.random.RandomState(0).rand(20, 3),
            ValueError,
            "further pair",
        ),
    ],
)
def test_invalid_input(params, X, err, match):
    with pytest.raises(err, match=match):
        PaCMAP(**params).fit(X)
