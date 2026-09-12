"""Scientific invariants for the CPU-only role-geometry supplement."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("scipy")
from scipy.spatial.distance import pdist, squareform

spec = importlib.util.spec_from_file_location(
    "role_geometry", Path(__file__).parents[1] / "scripts" / "postprocess_role_geometry.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def test_spread_separates_translation_from_compression():
    roles = np.random.default_rng(17).normal(size=(9, 5))
    translation = np.arange(5) * 10
    assert np.isclose(geometry.spread(roles + translation), geometry.spread(roles))
    assert np.isclose(geometry.spread(.5 * roles + translation) / geometry.spread(roles), .5)
    assert geometry.spread(np.ones((9, 5))) == 0


def test_shared_pca_preserves_known_low_rank_geometry():
    roles = np.array([[1, 0, 7], [-1, 0, 7], [0, 2, 7], [0, -2, 7]], dtype=float)
    center, basis, fractions = geometry.fit_pca(roles)
    coordinates = (roles - center) @ basis[:2].T
    assert np.isclose(fractions[:2].sum(), 1)
    assert np.allclose(pdist(coordinates), pdist(roles))
    assert np.allclose(coordinates @ basis[:2] + center, roles)


def test_clustering_distance_is_directional_not_effect_magnitude():
    vectors = np.array([[1, 0], [1, 1], [-1, 0]], dtype=float)
    similarity, unit = geometry.cosine_matrix(vectors)
    scaled, _ = geometry.cosine_matrix(vectors * np.array([2, 7, 4])[:, None])
    assert np.allclose(similarity, scaled)
    assert np.allclose(squareform(pdist(unit))**2, 2 - 2 * similarity)
    assert similarity[0, 2] == -1
