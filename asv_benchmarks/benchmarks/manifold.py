from sklearn.manifold import TSNE, PaCMAP

from .common import Benchmark, Estimator
from .datasets import _digits_dataset


class TSNEBenchmark(Estimator, Benchmark):
    """
    Benchmarks for t-SNE.
    """

    param_names = ["method"]
    params = (["exact", "barnes_hut"],)

    def setup_cache(self):
        super().setup_cache()

    def make_data(self, params):
        (method,) = params

        n_samples = 500 if method == "exact" else None

        return _digits_dataset(n_samples=n_samples)

    def make_estimator(self, params):
        (method,) = params

        estimator = TSNE(random_state=0, method=method)

        return estimator

    def make_scorers(self):
        self.train_scorer = lambda _, __: self.estimator.kl_divergence_
        self.test_scorer = lambda _, __: self.estimator.kl_divergence_


class PaCMAPBenchmark(Estimator, Benchmark):
    """
    Benchmarks for PaCMAP.
    """

    param_names = ["metric"]
    params = (["euclidean", "angular"],)

    def setup_cache(self):
        super().setup_cache()

    def make_data(self, params):
        return _digits_dataset()

    def make_estimator(self, params):
        (metric,) = params

        estimator = PaCMAP(random_state=0, metric=metric)

        return estimator

    def make_scorers(self):
        self.train_scorer = lambda _, __: self.estimator.loss_
        self.test_scorer = lambda _, __: self.estimator.loss_
