import numpy as np

import glotaran.optimization.nnls as nnls_module


def test_residual_nnls_doubles_default_maxiter(monkeypatch):
    calls: dict[str, int] = {}

    def fake_nnls(matrix, data, *, maxiter=None, atol=None):
        calls["maxiter"] = maxiter
        return np.array([1.0, 0.0]), 0.0

    monkeypatch.setattr(nnls_module, "nnls", fake_nnls)

    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    data = np.array([1.0, 2.0, 3.0])

    clp, residual = nnls_module.residual_nnls(matrix, data)

    assert calls["maxiter"] == 6 * matrix.shape[1]
    np.testing.assert_array_equal(clp, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(residual, data - matrix @ clp)
