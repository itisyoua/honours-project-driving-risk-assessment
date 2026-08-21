from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import torch


def validate_training_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ValueError("training manifest contains no rows")
    invalid_routes = sorted(
        {
            row.get("route_id", "<unknown>")
            for row in rows
            if row.get("split") != "train"
        }
    )
    if invalid_routes:
        preview = ", ".join(invalid_routes[:5])
        raise ValueError(
            f"training manifest must contain only train rows; found: {preview}"
        )


def source_balanced_sample_weights(
    rows: Sequence[Mapping[str, str]],
    default_source: str | None = None,
) -> torch.Tensor:
    if not rows:
        raise ValueError("cannot sample an empty manifest")
    sources = []
    for row in rows:
        source = row.get("source") or default_source
        if not source:
            raise ValueError("every manifest row must identify its source")
        sources.append(source)
    counts = Counter(sources)
    return torch.tensor([1.0 / counts[source] for source in sources], dtype=torch.double)
