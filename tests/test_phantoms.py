"""
Test UWCEM phantom loading and basic scene generation.

Requires UWCEM phantom data. Set WAVECARE_DATA_DIR env var or place
data in the default location (data/uwcem/ relative to project root).
"""

import os
import pytest
import numpy as np

from wavecare.synth.phantoms import load_uwcem_phantom, UWCEM_TISSUE_MAP

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get(
    "WAVECARE_DATA_DIR",
    os.path.join(_PROJECT_ROOT, "data", "uwcem"),
)

PHANTOM_IDS = ["071904", "062204"]


def _phantom_available(pid):
    return os.path.isdir(os.path.join(DATA_DIR, pid))


@pytest.fixture(params=[p for p in PHANTOM_IDS if _phantom_available(p)],
                ids=lambda p: f"phantom-{p}")
def phantom(request):
    """Load a UWCEM phantom (skipped if data not available)."""
    pid = request.param
    return load_uwcem_phantom(os.path.join(DATA_DIR, pid))


@pytest.mark.skipif(
    not any(_phantom_available(p) for p in PHANTOM_IDS),
    reason="No UWCEM phantom data available",
)
class TestLoadPhantom:
    def test_shape(self, phantom):
        mtype = phantom["mtype"]
        assert mtype.ndim == 3
        assert all(s > 0 for s in mtype.shape)

    def test_dtype(self, phantom):
        assert phantom["mtype"].dtype == np.int8

    def test_labels_in_range(self, phantom):
        labels = np.unique(phantom["mtype"])
        assert all(0 <= l <= 9 for l in labels)

    def test_voxel_size(self, phantom):
        assert phantom["voxel_mm"] == 0.5

    def test_has_breast_tissue(self, phantom):
        mtype = phantom["mtype"]
        # Should have at least background + skin + some interior tissue
        labels = set(np.unique(mtype))
        assert 0 in labels, "Missing background"
        assert 1 in labels, "Missing skin"
        assert len(labels) >= 4, "Too few tissue types"

    def test_tissue_map_keys(self, phantom):
        labels = set(np.unique(phantom["mtype"]))
        known = set(UWCEM_TISSUE_MAP.keys())
        assert labels.issubset(known), f"Unknown labels: {labels - known}"


@pytest.mark.skipif(
    not any(_phantom_available(p) for p in PHANTOM_IDS),
    reason="No UWCEM phantom data available",
)
class TestTumorInsertion:
    def test_insert_tumor(self, phantom):
        from wavecare.synth.tumors import (
            insert_tumor, random_tumor_position, random_tumor_params,
        )

        mtype = phantom["mtype"].copy()
        voxel_mm = phantom["voxel_mm"]
        rng = np.random.default_rng(42)

        params = random_tumor_params(rng=rng, diameter_range_mm=(10.0, 20.0))
        center = random_tumor_position(mtype, voxel_mm, margin_mm=5.0, rng=rng)
        params["center_mm"] = center

        mtype_with_tumor, n_vox = insert_tumor(
            mtype, voxel_mm, params, tumor_label=10, rng=rng,
        )

        assert 10 in mtype_with_tumor, "Tumor label 10 not found"
        assert n_vox > 0, "No tumor voxels inserted"

    def test_tumor_only_adds_label_10(self, phantom):
        from wavecare.synth.tumors import (
            insert_tumor, random_tumor_position, random_tumor_params,
        )

        mtype = phantom["mtype"].copy()
        voxel_mm = phantom["voxel_mm"]
        rng = np.random.default_rng(42)

        params = random_tumor_params(rng=rng, diameter_range_mm=(10.0, 20.0))
        center = random_tumor_position(mtype, voxel_mm, margin_mm=5.0, rng=rng)
        params["center_mm"] = center

        mtype_with_tumor, n_vox = insert_tumor(
            mtype, voxel_mm, params, tumor_label=10, rng=rng,
        )

        # Only difference should be voxels changed to label 10
        diff_mask = mtype_with_tumor != phantom["mtype"]
        assert np.all(mtype_with_tumor[diff_mask] == 10)
