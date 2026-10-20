# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Geometry-based "even" view sampling helpers.

Vendored unchanged from methods/TokenGS/tokengs/data/view_sampling.py; used
here by ffgs_images.py's _even_sample_context_and_target. In TokenGS it is
used by ``Provider._get_indices_static`` when ``opt.view_sampling == "even"``.
The idea: partition a scene's camera view-directions (points on a sphere) into
``k`` spatial clusters via k-means, then pick one random view per cluster each
iteration -- clusters guarantee roughly-even sphere coverage, the per-cluster
random pick gives variety across iterations.

Numpy-only on purpose: this runs inside forked DataLoader workers, so it must
import cheaply and carry no torch/scipy/sklearn dependency. Determinism is
controlled entirely by the ``seed`` argument to :func:`kmeans_unit` so that all
workers produce identical clusters.
"""

import numpy as np


def directions_from_c2ws(c2ws: np.ndarray) -> np.ndarray:
    """Unit view-direction vectors from camera-to-world matrices.

    Args:
        c2ws: ``(N, 4, 4)`` camera-to-world matrices. Camera centers are
            ``c2ws[:, :3, 3]`` (same convention as
            ``Provider._normalize_camera_mean_cam`` and the output of
            ``DL3DV10K.load_cameras``).

    Returns:
        ``(N, 3)`` unit vectors of ``(center - centroid_of_centers)``. For a
        camera rig sitting on a sphere this is the per-camera direction from the
        volume center, which is what we cluster for even angular coverage.
    """
    centers = np.asarray(c2ws)[:, :3, 3].astype(np.float64)
    d = centers - centers.mean(axis=0, keepdims=True)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)  # guard against a degenerate (coincident) center
    return d / n


def direction_signature(dirs: np.ndarray, k: int, decimals: int = 4) -> bytes:
    """Stable cache key for a (rig, k) pair.

    Rounds the direction vectors and hashes their bytes together with ``k``.
    Scenes that share an identical rig (e.g. every CQ500 scene today, matched to
    0.000 deg) yield the same signature -> clustered once. A future re-render
    with a per-scene randomly-rotated sphere yields a different signature ->
    re-clustered automatically, with no code change.

    Note: order-sensitive (bytes over the array as given). CQ500 rigs share the
    same frame ordering, so this is correct and cheapest. If a future rig can
    appear in different frame orderings but should share clusters, sort rows
    before hashing.
    """
    rounded = np.round(np.ascontiguousarray(dirs, dtype=np.float64), decimals)
    return rounded.tobytes() + int(k).to_bytes(4, "little")


def kmeans_unit(dirs: np.ndarray, k: int, seed: int, n_iter: int = 50) -> np.ndarray:
    """Euclidean Lloyd's k-means on (unit) direction vectors.

    Chordal (Euclidean) distance on unit vectors is monotonic with angular
    distance, which is all we need for even coverage. Deterministic given
    ``seed`` (farthest-point init + first-index tie-breaks), so all forked
    workers produce identical clusters. Every returned label class is guaranteed
    non-empty (empty-cluster reassignment), so callers can safely pick one member
    per cluster.

    Args:
        dirs: ``(N, 3)`` vectors, ``1 <= k <= N``.
        k: number of clusters.
        seed: RNG seed for the (only) random step, the init's first center.
        n_iter: maximum Lloyd iterations (early-breaks on stable labels).

    Returns:
        ``labels``: ``(N,)`` int64 in ``[0, k)``.
    """
    dirs = np.ascontiguousarray(dirs, dtype=np.float64)
    N = dirs.shape[0]
    assert 1 <= k <= N, f"kmeans_unit requires 1 <= k({k}) <= N({N})"
    if k == 1:
        return np.zeros(N, dtype=np.int64)
    if k == N:
        return np.arange(N, dtype=np.int64)

    # --- deterministic farthest-point init (greedy k-means++ / D2, seeded) ---
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, N))
    seed_idx = [first]
    d2 = np.sum((dirs - dirs[first]) ** 2, axis=1)  # min sq-dist to chosen centers
    for _ in range(1, k):
        nxt = int(np.argmax(d2))
        seed_idx.append(nxt)
        d2 = np.minimum(d2, np.sum((dirs - dirs[nxt]) ** 2, axis=1))
    centers = dirs[seed_idx].copy()

    labels = np.full(N, -1, dtype=np.int64)
    for _ in range(n_iter):
        dist = np.linalg.norm(dirs[:, None, :] - centers[None, :, :], axis=2)  # (N, k)
        new_labels = np.argmin(dist, axis=1)

        # Empty-cluster reassignment: move each empty center onto the currently
        # worst-served point, so all k clusters end up non-empty. Bounded loop.
        for _repair in range(k):
            empty = np.where(np.bincount(new_labels, minlength=k) == 0)[0]
            if empty.size == 0:
                break
            served = dist[np.arange(N), new_labels].copy()
            for c in empty:
                worst = int(np.argmax(served))
                centers[c] = dirs[worst]
                served[worst] = -1.0  # don't reuse the same point for two empties
            dist = np.linalg.norm(dirs[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dist, axis=1)

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            members = dirs[labels == c]
            if members.shape[0] > 0:
                centers[c] = members.mean(axis=0)

    return labels.astype(np.int64)


def labels_to_members(labels: np.ndarray, k: int) -> list[np.ndarray]:
    """Group indices by cluster label. Returns a length-``k`` list of index
    arrays; each is non-empty when ``labels`` comes from :func:`kmeans_unit`."""
    return [np.flatnonzero(labels == c) for c in range(k)]
