#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def short_sid(s):
    s = (s or '').strip()
    if s == 'path_open':
        return 'O'
    if s == 'path_covered':
        return 'C'
    return '.'


def fmt(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return '.'


def load_rows(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl_path')
    ap.add_argument('--trial-min', type=int, default=None)
    ap.add_argument('--trial-max', type=int, default=None)
    ap.add_argument('--real-junction-only', action='store_true')
    ap.add_argument('--show-all-post-shot', action='store_true')
    args = ap.parse_args()

    rows = load_rows(args.jsonl_path)

    out = []
    post_shot_seen = False
    for r in rows:
        tr = int(r.get('trial', -1))
        if args.trial_min is not None and tr < args.trial_min:
            continue
        if args.trial_max is not None and tr > args.trial_max:
            continue
        if args.real_junction_only and not bool(r.get('real_junction_choice_row', False)):
            continue

        if bool(r.get('step_one_shot_from_pending', False)):
            post_shot_seen = True

        if not args.show_all_post_shot and not post_shot_seen:
            continue

        qpl = r.get('q_pos_local', {}) or {}
        hol = r.get('h_opp_local', {}) or {}
        lb = r.get('local_bonus_values', []) or []
        qwb = r.get('q_values_with_local_bonus', []) or []
        probs = r.get('action_probs', []) or []

        row = {
            'tr': tr,
            'tk': int(r.get('tick', -1)),
            'cfg': short_sid(r.get('configured_one_shot_source_id', '')),
            'actv': short_sid(r.get('active_one_shot_source_id', '')),
            'obs': short_sid(r.get('obs_pre_one_shot_source_id', '')),
            'cand': short_sid(r.get('candidate_source_id', '')),
            'com': short_sid(r.get('committed_source_id', '')),
            'shot': 'Y' if bool(r.get('step_one_shot_from_pending', False)) else '.',
            'qpo': fmt(qpl.get('path_open', 0.0)),
            'qpc': fmt(qpl.get('path_covered', 0.0)),
            'hopo': fmt(hol.get('path_open', 0.0)),
            'hopc': fmt(hol.get('path_covered', 0.0)),
            'lbo': fmt(lb[0] if len(lb) > 0 else 0.0),
            'lbc': fmt(lb[1] if len(lb) > 1 else 0.0),
            'qbo': fmt(qwb[0] if len(qwb) > 0 else 0.0),
            'qbc': fmt(qwb[1] if len(qwb) > 1 else 0.0),
            'po': fmt(probs[0] if len(probs) > 0 else 0.0),
            'pc': fmt(probs[1] if len(probs) > 1 else 0.0),
            'act': 'O' if int(r.get('action', -1)) == 0 else ('C' if int(r.get('action', -1)) == 1 else '.'),
            'commit': 'Y' if bool(r.get('post_deliberation_state') == 'committed') else '.',
        }
        out.append(row)

    cols = ['tr','tk','cfg','actv','obs','cand','com','shot','qpo','qpc','hopo','hopc','lbo','lbc','qbo','qbc','po','pc','act','commit']
    widths = {c: max(len(c), max((len(str(r[c])) for r in out), default=0)) for c in cols}
    header = '  '.join(c.rjust(widths[c]) for c in cols)
    print(header)
    print('-' * len(header))
    for r in out:
        print('  '.join(str(r[c]).rjust(widths[c]) for c in cols))

    if out:
        cfg = out[0]['cfg']
        target_col = 'pc' if cfg == 'C' else 'po'
        target_lb = 'lbc' if cfg == 'C' else 'lbo'
        target_h = 'hopc' if cfg == 'C' else 'hopo'
        target_q = 'qpc' if cfg == 'C' else 'qpo'
        target_prob_wins = sum(float(r[target_col]) >= 0.5 for r in out)
        target_lb_pos = sum(float(r[target_lb]) > 0.0 for r in out)
        target_h_pos = sum(float(r[target_h]) > 0.0 for r in out)
        target_q_pos = sum(float(r[target_q]) > 0.0 for r in out)
        print('\nSummary')
        print('-------')
        print(f'rows={len(out)} cfg={cfg} target_q>0={target_q_pos}/{len(out)} target_h>0={target_h_pos}/{len(out)} target_lb>0={target_lb_pos}/{len(out)}')
        print(f'target_prob_wins={target_prob_wins}/{len(out)}')


if __name__ == '__main__':
    main()
