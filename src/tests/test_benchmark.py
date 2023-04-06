import numpy as np
import pytest

from nn_vs_ann import benchmark


@pytest.mark.parametrize(
    # fmt: off
    "k, expected", [
        (1, [2]), (2, [2, 1]), (3, [2, 1, 0])
    ]
    # fmt: on
)
def test_topk(k, expected):
    # fmt: off
    mat = np.array(
        [   
            [1, 3, 1], 
            [1, 2, 1], 
            [1, 1, 1], 
            [1, 4, 1]
        ], dtype=np.float32
    )
    # fmt: on
    vec = np.array([1, 1, 1], dtype=np.float32)
    expected = np.array(expected)
    res, _ = benchmark.topk(vec, mat, k=k, do_norm=True)
    np.testing.assert_array_equal(expected, res)
