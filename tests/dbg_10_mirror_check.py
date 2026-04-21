#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def f3(x: Any) -> str:
    if x is None:
        return '.'
    try:
        v = float(x)
    except (TypeError, ValueError):
        return '.'
    if math.isnan(v):
        return '.'
    return f"{v:.3f}"


def path_label(x: Any) -> str:
    s = '' if x is None else str(x)
    if s == 'path_open' or s == 'open':
        return 'O'
    if s == 'path_covered' or s == 'covered':
        return 'C'
    if s in ('', 'None', 'null'):
        return '.'
    return s


def get_local(d: Any, key: str) -> float:
    if not isinstance(d, dict):
        return 0.0
    try:
        return float(d.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def get_idx(xs: Any, idx: int) -> float:
    if not isinstance(xs, list) or idx >= len(xs):
        return 0.0
    try:
        return float(xs[idx])
    except (TypeError, ValueError):
        return 0.0


def yesno(x: bool) -> str:
    return 'Y' if bool(x) else '.'


def build_behavior(row: Dict[str, Any]) -> str:
    cand = path_label(row.get('candidate_source_id') or row.get('candidate_path'))
    com = path_label(row.get('committed_source_id') or row.get('committed_path'))
    act = row.get('action')
    try:
        act_s = 'O' if int(act) == 0 else 'C'
    except Exception:
        act_s = '.'

    state = str(row.get('post_deliberation_state', row.get('pre_deliberation_state', '')) or '')
    if state == 'committed':
        return f"{act_s}:{cand}->{com}*"
    if state == 'traversing':
        return f"{act_s}:{cand}->{com}"
    return f"{act_s}:{cand}->."


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Compact mirror-check view for Stage 3.1B one-shot source-specific carryover.'
    )
    ap.add_argument('path', help='Path to *_debug_trace.jsonl')
    ap.add_argument('--trial-min', type=int, default=30)
    ap.add_argument('--trial-max', type=int, default=32)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--real-junction-only', action='store_true')
    args = ap.parse_args()

    rows = load_jsonl(Path(args.path))
    if args.seed is not None:
        rows = [r for r in rows if int(r.get('seed', -1)) == args.seed]
    rows = [r for r in rows if args.trial_min <= int(r.get('trial', -1)) <= args.trial_max]
    if args.real_junction_only:
        rows = [r for r in rows if bool(r.get('real_junction_choice_row', False))]

    if not rows:
        print('No rows matched filter.')
        return

    headers = [
        'tr', 'tk', 'row', 'cfg', 'obs', 'cand', 'com', 'bhv',
        'qpo', 'qpc', 'hopo', 'hopc', 'lbo', 'lbc', 'qbo', 'qbc',
        'po', 'pc', 'act', 'commit', 'forced', 'shot'
    ]
    print(' '.join(f"{h:>7}" for h in headers))
    print('-' * (8 * len(headers)))

    for r in rows:
        qpl = r.get('q_pos_local', {})
        hol = r.get('h_opp_local', {})
        lb = r.get('local_bonus_values', [])
        qb = r.get('q_values_with_local_bonus', [])
        probs = r.get('action_probs', [])

        qpo = get_local(qpl, 'path_open')
        qpc = get_local(qpl, 'path_covered')
        hopo = get_local(hol, 'path_open')
        hopc = get_local(hol, 'path_covered')
        lbo = get_idx(lb, 0)
        lbc = get_idx(lb, 1)
        qbo = get_idx(qb, 0)
        qbc = get_idx(qb, 1)
        po = get_idx(probs, 0)
        pc = get_idx(probs, 1)

        pre_j = bool(r.get('pre_at_junction', False))
        row_kind = 'J' if pre_j else '.'
        if bool(r.get('real_junction_choice_row', False)):
            row_kind = 'RJ'

        act = r.get('action')
        act_s = '.'
        try:
            act_s = 'O' if int(act) == 0 else 'C'
        except Exception:
            pass

        committed = str(r.get('post_deliberation_state', '')) == 'committed'
        shot = bool(r.get('step_one_shot_from_pending', False)) or str(r.get('one_shot_type', 'none')) != 'none'

        vals = [
            str(r.get('trial', '.')),
            str(r.get('tick', '.')),
            row_kind,
            path_label(r.get('configured_one_shot_source_id')),
            path_label(r.get('obs_pre_one_shot_source_id')),
            path_label(r.get('candidate_source_id') or r.get('candidate_path')),
            path_label(r.get('committed_source_id') or r.get('committed_path')),
            build_behavior(r),
            f3(qpo), f3(qpc), f3(hopo), f3(hopc),
            f3(lbo), f3(lbc), f3(qbo), f3(qbc),
            f3(po), f3(pc),
            act_s,
            yesno(committed),
            yesno(r.get('forced_action_applied', False)),
            yesno(shot),
        ]
        print(' '.join(f"{v:>7}" for v in vals))

    mirror_rows = [r for r in rows if bool(r.get('real_junction_choice_row', False))]
    if mirror_rows:
        cfg_ids = {str(r.get('configured_one_shot_source_id', '')) for r in mirror_rows}
        cfg = next(iter(cfg_ids)) if len(cfg_ids) == 1 else 'mixed'
        target = 'path_covered' if cfg == 'path_covered' else 'path_open' if cfg == 'path_open' else None
        other = 'path_open' if target == 'path_covered' else 'path_covered' if target == 'path_open' else None

        target_lb_positive = 0
        other_lb_positive = 0
        target_prob_wins = 0
        commits_to_target = 0
        for r in mirror_rows:
            lb = r.get('local_bonus_values', [])
            probs = r.get('action_probs', [])
            cand_sid = str(r.get('committed_source_id') or r.get('candidate_source_id') or '')

            if target == 'path_open':
                tlb = get_idx(lb, 0)
                olb = get_idx(lb, 1)
                tp = get_idx(probs, 0)
                op = get_idx(probs, 1)
            elif target == 'path_covered':
                tlb = get_idx(lb, 1)
                olb = get_idx(lb, 0)
                tp = get_idx(probs, 1)
                op = get_idx(probs, 0)
            else:
                tlb = olb = tp = op = 0.0

            if tlb > 1e-12:
                target_lb_positive += 1
            if olb > 1e-12:
                other_lb_positive += 1
            if tp > op:
                target_prob_wins += 1
            if cand_sid == target:
                commits_to_target += 1

        n = len(mirror_rows)
        print('\nSummary')
        print('-------')
        print(f'rows={n} cfg={path_label(cfg)} target_lb>0={target_lb_positive}/{n} other_lb>0={other_lb_positive}/{n}')
        print(f'target_prob_wins={target_prob_wins}/{n} target_choice_or_commit={commits_to_target}/{n}')
        if target is not None:
            print('mirror_ok=' + ('YES' if (target_lb_positive > 0 and other_lb_positive == 0) else 'NO'))


if __name__ == '__main__':
    main()
