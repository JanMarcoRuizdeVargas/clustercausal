from clustercausal.utils.Utils import *
import numpy as np


def test_causallearn_to_nx_adjmat_and_back():
    causallearn_adjmat = np.array(
        [
            [0, 0, 0, 1, 0, -1, 0, -1, -1, 0],
            [0, 0, 1, 1, 0, -1, 1, -1, -1, -1],
            [0, -1, 0, 0, -1, 0, -1, 0, 0, 0],
            [-1, -1, 0, 0, 0, -1, 0, -1, -1, 0],
            [0, 0, 1, 0, 0, 0, 1, -1, 0, 0],
            [1, 1, 0, 1, 0, 0, 1, 0, 0, -1],
            [0, -1, -1, 0, -1, -1, 0, -1, -1, 0],
            [1, 1, 0, 1, 1, 0, 1, 0, 1, -1],
            [1, 1, 0, 1, 0, 0, 1, -1, 0, -1],
            [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        ]
    )
    nx_adjmat = causallearn_to_nx_adjmat(causallearn_adjmat)
    causallearn_adjmat_2 = nx_to_causallearn_adjmat(nx_adjmat)
    assert np.array_equal(causallearn_adjmat, causallearn_adjmat_2)
