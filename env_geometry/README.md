# Environment Geometry Registry

This package contains static schematic geometry for Stage 2 and Stage 3 tasks.

It is a service layer, not a simulator. It must not import `stage2/`, `stage3/`,
or `vte/` runtime internals.

## Purpose

The registry provides:

- task-topology schematics for external readers;
- a common vocabulary for choice points, routes, commit zones, reward zones, and event-aligned windows;
- a future bridge to biological-lab VTE adapters.

## Current built-in geometries

- `stage2_twostep_daw2011`
- `stage2_reversal`
- `stage3_open_covered_choice`

The coordinates are schematic. They are not physical lab coordinates.

## Render static schematics

```bash
python -m env_geometry.render_static --output-dir logs/env_geometry
```

List available geometries:

```bash
python -m env_geometry.render_static --list
```

Render one geometry:

```bash
python -m env_geometry.render_static \
  --env-id stage3_open_covered_choice \
  --output-dir logs/env_geometry
```

## Outputs

```text
logs/env_geometry/
├── Figure_EnvGeometry_stage2_twostep_daw2011.png
├── Figure_EnvGeometry_stage2_reversal.png
├── Figure_EnvGeometry_stage3_open_covered_choice.png
├── env_geometry_registry_export.json
└── env_geometry_render_manifest.json
```

## Boundary

The geometry registry is descriptive. It does not define model dynamics, reward
updates, gate behavior, VTE labels, or biological equivalence.