from __future__ import annotations

import pytest

from scripts.hrc.plot_dissertation_results import require_columns


def test_require_columns_rejects_incomplete_frozen_csv() -> None:
    with pytest.raises(ValueError, match="missing columns.*std"):
        require_columns([{"model": "A20", "mean": "0.1"}], {"model", "mean", "std"}, "audit")
