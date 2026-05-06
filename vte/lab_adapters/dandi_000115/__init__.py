"""DANDI 000115 lab adapter probes.

Patch 15A scope:
- inspect local NWB behavior layer;
- extract Position / SpatialSeries summaries;
- extract behavioral event channel summaries;
- extract StateScript logs stored as AssociatedFiles attributes;
- compute diagnostic StateScript-to-position alignment candidates.

This package does not claim direct biological VTE labels.
"""
