#!/usr/bin/env python3
"""
Validate Stage 3.1B one-shot acceptance summary.

This is a statistical / interpretive guard, not an artifact-existence guard.

Smoke mode:
- requires schema pass;
- requires post_all and post_11_30 to have the expected direction;
- allows underpowered directional-only evidence;
- requires placebo-null rows to exist;
- requires shock placebo-null to pass;
- allows treat placebo-null to remain diagnostic;
- requires protocol-specific carrier checks:
  shock through h_risk/q_neg, treat through h_opp/q_pos;
- does not require post_31_plus to pass.

Full mode:
- requires schema pass;
- requires post_all and post_11_30 to be full pass for the selected ablation;
- requires shock placebo-null to pass;
- allows treat placebo-null to remain diagnostic;
- requires at least one protocol-specific carrier metric to fully pass
  for each core carrier window;
- still treats post_31_plus as diagnostic only.

Rationale:
Stage 3.1B closure should show persistent one-shot deformation over tens of
trials, not permanent infinite persistence. Therefore post_11_30 and post_all
are core; post_31_plus is informative but not a hard acceptance criterion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


CORE_EFFECT_CHECKS = {
    "p_target_delta_post_11_30",
    "p_target_delta_post_all",
}

DIAGNOSTIC_ONLY_CHECKS = {
    "p_target_delta_post_31_plus",
}

FIRST_POST_CHECKS = {
    "first_post_target_probability",
}

SCHEMA_CHECKS = {
    "schema_required_columns",
}

PLACEBO_CORE_CHECKS = {
    "placebo_window_null_post_11_30",
    "placebo_window_null_post_all",
}

CARRIER_CORE_WINDOWS = {
    "post_1_3",
    "post_4_10",
    "post_11_30",
}

CARRIER_METRICS_BY_PROTOCOL = {
    "shock": {"h_risk", "q_neg"},
    "treat": {"h_opp", "q_pos"},
}

EXPECTED_FULL_ABLATION_COMPARISONS = {
    "novg",
    "novp",
    "nox",
    "one_shot_off",
}

ABLATION_LOCALIZATION_PREFIX = "ablation_localization_"
ABLATION_LOCALIZATION_AVAILABILITY_CHECK = "ablation_localization_available"

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate Stage 3.1B one-shot acceptance summary")
    ap.add_argument(
        "--results-root",
        default="docs/results",
        help="Curated results root containing stage3_1b_closure",
    )
    ap.add_argument(
        "--acceptance-csv",
        default=None,
        help="Optional direct path to Table_3_1B_one_shot_acceptance_summary.csv",
    )
    ap.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Validation strictness",
    )
    ap.add_argument(
        "--ablation",
        default="full",
        help="Ablation to validate as the main claim carrier",
    )
    return ap.parse_args()


def acceptance_path(args: argparse.Namespace) -> Path:
    if args.acceptance_csv:
        return Path(args.acceptance_csv)
    return (
        Path(args.results_root)
        / "stage3_1b_closure"
        / "tables"
        / "Table_3_1B_one_shot_acceptance_summary.csv"
    )


def is_direction_ok(status: str) -> bool:
    return status in {"pass", "direction_ok"}


def is_full_pass(status: str) -> bool:
    return status == "pass"

def carrier_check_names_for_protocol(protocol: str) -> List[str]:
    metrics = CARRIER_METRICS_BY_PROTOCOL.get(protocol, set())
    out: List[str] = []
    for metric in sorted(metrics):
        for window in sorted(CARRIER_CORE_WINDOWS):
            out.append(f"carrier_delta_{metric}_{window}")
    return out

def is_placebo_acceptable(protocol: str, status: str) -> bool:
    """
    Shock is expected to be event-boundary-specific, so placebo-null should pass.

    Treat may show a broader positive approach drift in short smoke runs; for
    treat, placebo-null is required to exist but diagnostic status is acceptable.
    Full-mode treat acceptance should rely on positive carrier/effect/ablation
    localization rather than requiring event-boundary specificity.
    """
    if protocol == "shock":
        return status == "pass"
    if protocol == "treat":
        return status in {"pass", "diagnostic"}
    return status in {"pass", "diagnostic"}

def validate(df: pd.DataFrame, mode: str, ablation: str) -> List[str]:
    errors: List[str] = []

    required_columns = {
        "protocol",
        "ablation",
        "check",
        "status",
        "value",
        "threshold_or_expectation",
        "note",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        return [f"acceptance summary is missing columns: {missing}"]

    main = df[df["ablation"].astype(str) == str(ablation)].copy()
    if main.empty:
        return [f"no acceptance rows found for ablation={ablation!r}"]

    expected_protocols = {"shock", "treat"}

    for protocol in expected_protocols:
        pdf = main[main["protocol"].astype(str) == protocol].copy()
        if pdf.empty:
            errors.append(f"missing protocol rows: protocol={protocol}, ablation={ablation}")
            continue

        schema = pdf[pdf["check"].isin(SCHEMA_CHECKS)]
        if schema.empty:
            errors.append(f"missing schema check: protocol={protocol}, ablation={ablation}")
        else:
            bad_schema = schema[schema["status"].astype(str) != "pass"]
            for _, row in bad_schema.iterrows():
                errors.append(
                    f"schema failed: protocol={protocol}, ablation={ablation}, "
                    f"status={row['status']}, note={row.get('note', '')}"
                )

        for check in CORE_EFFECT_CHECKS:
            rows = pdf[pdf["check"].astype(str) == check]
            if rows.empty:
                errors.append(f"missing core effect check: protocol={protocol}, check={check}")
                continue

            row = rows.iloc[0]
            status = str(row["status"])

            if mode == "smoke":
                if not is_direction_ok(status):
                    errors.append(
                        f"core effect lacks expected direction in smoke: "
                        f"protocol={protocol}, check={check}, status={status}, value={row['value']}"
                    )
            else:
                if not is_full_pass(status):
                    errors.append(
                        f"core effect is not a full pass in full mode: "
                        f"protocol={protocol}, check={check}, status={status}, "
                        f"value={row['value']}, note={row.get('note', '')}"
                    )

        first_post = pdf[pdf["check"].isin(FIRST_POST_CHECKS)]
        if first_post.empty:
            errors.append(f"missing first-post probability check: protocol={protocol}")
        else:
            for _, row in first_post.iterrows():
                status = str(row["status"])
                if not is_direction_ok(status):
                    errors.append(
                        f"first-post target probability has wrong direction: "
                        f"protocol={protocol}, status={status}, value={row['value']}, "
                        f"note={row.get('note', '')}"
                    )

        for check in PLACEBO_CORE_CHECKS:
            rows = pdf[pdf["check"].astype(str) == check]
            if rows.empty:
                errors.append(f"missing placebo-null check: protocol={protocol}, check={check}")
                continue

            row = rows.iloc[0]
            status = str(row["status"])

            if not is_placebo_acceptable(protocol, status):
                errors.append(
                    f"placebo-null check failed acceptance policy: "
                    f"protocol={protocol}, check={check}, status={status}, "
                    f"value={row['value']}, note={row.get('note', '')}"
                )

        carrier_checks = carrier_check_names_for_protocol(protocol)

        for check in carrier_checks:
            rows = pdf[pdf["check"].astype(str) == check]
            if rows.empty:
                errors.append(f"missing carrier check: protocol={protocol}, check={check}")
                continue

            row = rows.iloc[0]
            status = str(row["status"])

            if mode == "smoke":
                if not is_direction_ok(status):
                    errors.append(
                        f"carrier check lacks expected positive direction in smoke: "
                        f"protocol={protocol}, check={check}, status={status}, "
                        f"value={row['value']}, note={row.get('note', '')}"
                    )
            else:
                # Full mode is stricter but not overdetermined:
                # for each protocol/window, at least one carrier metric must be a full pass.
                pass

        if mode == "full":
            for window in sorted(CARRIER_CORE_WINDOWS):
                candidate_checks = [
                    f"carrier_delta_{metric}_{window}"
                    for metric in CARRIER_METRICS_BY_PROTOCOL.get(protocol, set())
                ]
                rows = pdf[pdf["check"].astype(str).isin(candidate_checks)]
                if rows.empty:
                    errors.append(
                        f"missing carrier window checks: protocol={protocol}, window={window}"
                    )
                    continue

                if not any(is_full_pass(str(status)) for status in rows["status"].tolist()):
                    errors.append(
                        f"no carrier metric is a full pass in full mode: "
                        f"protocol={protocol}, window={window}, "
                        f"statuses={rows[['check', 'status', 'value']].to_dict(orient='records')}"
                    )
    errors.extend(
        validate_ablation_localization_availability(
            df,
            mode=mode,
            ablation=ablation,
        )
    )
    return errors

def validate_ablation_localization_availability(
    df: pd.DataFrame,
    *,
    mode: str,
    ablation: str,
) -> List[str]:
    errors: List[str] = []

    main = df[df["ablation"].astype(str) == str(ablation)].copy()
    if main.empty:
        return [f"no rows found for ablation={ablation!r}"]

    availability = main[
        main["check"].astype(str) == ABLATION_LOCALIZATION_AVAILABILITY_CHECK
    ]

    localization_rows = main[
        main["check"].astype(str).str.startswith(ABLATION_LOCALIZATION_PREFIX)
        & (main["check"].astype(str) != ABLATION_LOCALIZATION_AVAILABILITY_CHECK)
    ].copy()

    if mode == "smoke":
        # Smoke normally runs --ablations full only. Availability diagnostic is allowed.
        return errors

    if len(availability):
        errors.append(
            "full mode requires non-full ablation localization rows; "
            "found only/also availability diagnostic placeholder"
        )

    if localization_rows.empty:
        errors.append(
            "full mode requires ablation-localization checks against non-full ablations"
        )
        return errors

    check_text = " ".join(localization_rows["check"].astype(str).tolist())

    missing = sorted(
        ablation_name
        for ablation_name in EXPECTED_FULL_ABLATION_COMPARISONS
        if f"_vs_{ablation_name}" not in check_text
    )

    if missing:
        errors.append(
            f"full mode missing ablation-localization comparisons: {missing}"
        )

    # Do not require every localization row to pass yet.
    # Some ablations may localize behavior, others carriers, and some are diagnostic.
    # The paper-grade full run should first expose the pattern.
    return errors

def print_compact(df: pd.DataFrame, ablation: str) -> None:
    cols = ["protocol", "ablation", "check", "status", "value", "note"]
    sdf = df[df["ablation"].astype(str) == str(ablation)].copy()
    if len(sdf):
        print(sdf[cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    path = acceptance_path(args)

    if not path.exists():
        print(f"ERROR: acceptance summary not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    print(f"Acceptance summary: {path}")
    print(f"Mode: {args.mode}")
    print(f"Ablation: {args.ablation}")
    print()
    print_compact(df, args.ablation)
    print()

    errors = validate(df, mode=args.mode, ablation=args.ablation)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"\nStage 3.1B acceptance validation failed: {len(errors)} error(s)")
        sys.exit(1)

    print("Stage 3.1B acceptance validation passed")


if __name__ == "__main__":
    main()