import json
import time

import hnswlib
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from nn_vs_ann import ROOT


def cosine_similarity(
    vec: npt.NDArray[np.float32], mat: npt.NDArray[np.float32], do_norm: bool = True
) -> npt.NDArray[np.float32]:
    """Calculate the cosine similarity between a vector and every row vector in a
    matrix.


    Parameters
    ----------
    vec
        A 1D array of shape D.
    mat
        A 2D array of shape N x D.
    do_norm, optional
        Whether to divide by the vector norms, by default True. Set this to
        False if the vectors are already normalized.

    Returns
    -------
    sim
        A 1D array of shape N. Each element `n` corresponds to the cosine similarity
        between `vec` and the vector at row `n` in `mat`.
    """
    sim = vec @ mat.T
    if do_norm:
        sim /= np.linalg.norm(vec) * np.linalg.norm(mat, axis=1)
    return sim


def topk(
    vec: npt.NDArray[np.float32],
    mat: npt.NDArray[np.float32],
    k: int = 5,
    do_norm: bool = True,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Given a vector, find the top K closest vectors in a matrix of vectors.


    Parameters
    ----------
    vec
        A 1D array of shape D.
    mat
        A 2D array of shape N x D.
    k, optional
        The number of closest vectors to find., by default 5
    do_norm, optional
        Whether to divide by the vector norms, by default True. Set this to
        False if the vectors are already normalized.

    Returns
    -------
    indices
        A 1D array of shape `k` corresponding to the row indices of mat with the closest
        vectors to `vec`.
    sim
        A 1D array the same shape as `indices` containing the cosine similarity scores
        for each element in `indices`.
    """
    sim = cosine_similarity(vec, mat, do_norm=do_norm)
    # Rather than sorting all similarities and taking the top K, it's faster to
    # argpartition and then just sort the top K.
    # The difference is O(N logN) vs O(N + k logk)
    indices = np.argpartition(-sim, kth=k)[:k]
    top_indices = np.argsort(-sim[indices])
    return indices[top_indices], sim[top_indices]


def knn(
    vec: npt.NDArray[np.float32], clf
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    sims, indices = clf.kneighbors(vec.reshape(1, -1))
    return indices, sims


@numba.njit("float32[:,:](int_, int_)", parallel=True)
def fast_random_array(n: int, m: int) -> npt.NDArray[np.float32]:
    """A numba-optimized function to quickly generate a random 2D array.

    The implementation was taken from https://stackoverflow.com/a/70443846

    Parameters
    ----------
    n
        The number of rows in the random array.
    m
        The number of columns in the random array.

    Returns
    -------
    res
        A 2D array of random numbers.
    """
    res = np.empty((n, m), dtype=np.float32)

    for i in numba.prange(n):
        for j in range(m):
            res[i, j] = np.random.rand()

    return res


def main(
    num_dims: int,
    k: int,
    num_trials: int,
    size_range: list[int],
) -> pd.DataFrame:
    """Run the benchmarks.

    Parameters
    ----------
    num_dims
        The number of dimensions each embedding should have.
    k
        The number of closest embeddings to search for.
    num_trials
        The number of trials to run for each benchmark.
    size_range
        A list containing the numbers of embeddings to run benchmarks for, such as
        [1_000, 10_000] to run benchmarks for 1,000 embeddings and 10,000 embeddings.

    Returns
    -------
    results
        A DataFrame containing results from all benchmarks.
    """
    results = []
    for num_embeddings in size_range:
        print(
            f"Creating sample data of {num_embeddings:,} {num_dims}-dimensional vectors"
        )
        data = fast_random_array(num_embeddings, num_dims)

        #################################################
        #                 hnswlib                       #
        #################################################
        print("Begin hnswlib benchmark.")
        print("Building index...")
        t0 = time.time()

        # NOTE: Much of the below hnswlib code was just taken from the hnswlib README.

        # NOTE: We choose inner product since this is equivalent to cosine if you
        # normalize vectors prior to insertion.
        p = hnswlib.Index(
            space="ip", dim=num_dims
        )  # possible options are l2, cosine or ip

        # NOTE: Use default settings from the README.
        p.init_index(max_elements=num_embeddings, ef_construction=200, M=16)
        ids = np.arange(num_embeddings)
        p.add_items(data, ids)
        p.set_ef(50)

        t1 = time.time()
        print(f"Took {t1 - t0:.3f}s to build index.")

        print("Starting hnswlib trials...")
        times = []
        for i in range(num_trials):
            t0 = time.time()
            p.knn_query(data[[i], :], k=k)
            t1 = time.time()
            times.append(t1 - t0)

        results.append(
            {
                "num_embeddings": num_embeddings,
                "num_dims": num_dims,
                "lib": "hnswlib",
                "k": k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )
        print(json.dumps(results[-1], indent=2))

        print("Done hnswlib")

        #################################################
        #                 numpy                         #
        #################################################

        print("Starting numpy trials...")
        times = []
        for i in range(num_trials):
            t0 = time.time()
            # NOTE: Do not normalize so that similarity is only the inner product in
            # order to match hnswlib.
            topk(data[i], data, k=k, do_norm=False)
            t1 = time.time()
            times.append(t1 - t0)

        results.append(
            {
                "num_embeddings": num_embeddings,
                "num_dims": num_dims,
                "lib": "numpy",
                "k": k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )
        print(json.dumps(results[-1], indent=2))

        print("Done numpy.")

        #################################################
        #                 KNN                           #
        #################################################

        print("Starting KNN trials...")

        t0 = time.time()
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(data, np.arange(len(data)))
        t1 = time.time()
        print(f"Took {t1 - t0:.3f}s to build KNN.")
        times = []
        for i in range(num_trials):
            t0 = time.time()
            knn(data[i], clf)
            t1 = time.time()
            times.append(t1 - t0)

        results.append(
            {
                "num_embeddings": num_embeddings,
                "num_dims": num_dims,
                "lib": "knn",
                "k": k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )
        print(json.dumps(results[-1], indent=2))

        print("Done KNN.")

    results = pd.DataFrame(results)
    return results


if __name__ == "__main__":
    # Benchmark Config
    num_dims = 256
    num_trials = 25
    size_range = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    k = 10

    results = main(num_dims, k, num_trials, size_range)
    print(results)
    results.to_csv(ROOT / "assets" / "results.csv", index=False)
