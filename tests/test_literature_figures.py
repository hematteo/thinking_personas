"""Scientific and text-alignment invariants for the literature adaptations."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tokenizers")
spec = importlib.util.spec_from_file_location("literature_figures", Path(__file__).parents[1] / "scripts/build_literature_figures.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_scree_distinguishes_compression_from_dimensionality():
    points = np.array([[2., 0., 0.], [-2., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
    _, basis, variance = module.pca(points)
    _, _, compressed = module.pca(points * .4 + np.array([10., 20., 30.]))
    assert np.allclose(variance, [2., .5])
    assert np.allclose(compressed, variance * .16)
    assert np.allclose(compressed/compressed.sum(), variance/variance.sum())
    assert np.allclose((points @ basis.T) @ basis, points)


def test_position_bins_preserve_counts_and_weighted_mean():
    for size in [3, 101, 217]:
        values = np.arange(size, dtype=float) - 32
        binned, counts = module.bin_tokens(values)
        assert counts.sum() == size
        assert np.isnan(binned).sum() == (counts == 0).sum()
        assert np.isclose(np.nansum(binned*counts)/counts.sum(), values.mean())


def test_unicode_grouping_keeps_token_scores_and_text():
    inverse = {value: key for key, value in module.byte_decoder().items()}
    assert sorted(inverse) == list(range(256))
    class ByteTokenizer:
        def id_to_token(self, token_id):
            return inverse[token_id]
        def decode(self, ids, skip_special_tokens=False):
            return bytes(ids).decode("utf-8")
    text = ' A naïve “poet” <script>\n🌍'
    ids = list(text.encode("utf-8"))
    groups = module.text_groups(ids, range(len(ids)), np.arange(len(ids))*.25, ByteTokenizer())
    assert "".join(g["text"] for g in groups) == text
    assert [i for g in groups for i in g["ids"]] == ids
    assert [i for g in groups for i in g["indices"]] == list(range(len(ids)))
    assert any(len(g["ids"]) == 4 for g in groups)
