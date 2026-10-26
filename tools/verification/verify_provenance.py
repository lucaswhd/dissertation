#!/usr/bin/env python3
"""Read-only final HRC provenance verification. All writes stay in --output-root.

Example (existing environment, no packages installed):
/root/autodl-tmp/openpi_hrc/.venv/bin/python \
 /root/dissertation_evidence_verification/scripts/verify_provenance.py \
 --batch-root /root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_rerun_v1 \
 --dataset-root /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_rerun_v1 \
 --pipeline-root /root/autodl-tmp/hrc_pipeline_traintest \
 --output-root /root/dissertation_evidence_verification

Hashes every referenced retained raw file (not just equal-size candidates), and
all retained HRC HDF5 files. Separately hashes canonical numeric action/state
arrays to detect identical demonstrations in containers with different metadata.
Does not train, simulate, decode video, or alter original files.
"""
import argparse
import ast
import collections
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import h5py
import numpy as np
import pyarrow.parquet as pq


def plain(v):
    if isinstance(v, dict): return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, bytes): return v.decode('utf-8', errors='replace')
    if isinstance(v, Path): return str(v)
    return v


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def arrinfo(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(json.dumps({'dtype': a.dtype.str, 'shape': list(a.shape)}, sort_keys=True).encode())
    h.update(b'\n')
    h.update(a.tobytes(order='C'))
    return {'shape': list(a.shape), 'dtype': a.dtype.str, 'canonical_sha256': h.hexdigest()}


def readjson(p):
    return json.loads(Path(p).read_text())


def group_dups(rows, key):
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get(key) is not None: groups[str(r[key])].append(r)
    return [{'identity': k, 'members': [{'source_id': x['source_id'], 'split': x['split'], 'stage': x['stage'], 'path': x['path']} for x in v]} for k, v in groups.items() if len(v) > 1]


def git_read(root, *args):
    p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    return {'exit_code': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch-root', type=Path, required=True)
    ap.add_argument('--dataset-root', type=Path, required=True)
    ap.add_argument('--pipeline-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_root.resolve()
    if out != Path('/root/dissertation_evidence_verification'):
        raise SystemExit('Refusing output outside the expressly authorized delivery root')
    for sub in ['reports', 'evidence']: (out / sub).mkdir(parents=True, exist_ok=True)
    started = time.time()
    evidence = {'run': {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': sys.argv, 'executable': sys.executable, 'python': platform.python_version(), 'packages': {x: importlib.metadata.version(x) for x in ['numpy', 'h5py', 'pyarrow']}, 'read_only_originals': True}, 'files': {}, 'issues': []}
    def record(p):
        p = Path(p)
        k = str(p.absolute())
        if k in evidence['files']: return evidence['files'][k]
        r = {'path': k, 'exists': p.exists(), 'is_symlink': p.is_symlink(), 'realpath': str(p.resolve())}
        if p.exists():
            s = p.stat()
            r.update({'size_bytes': s.st_size, 'device': s.st_dev, 'inode': s.st_ino, 'mtime_ns': s.st_mtime_ns, 'sha256': sha(p)})
        evidence['files'][k] = r
        return r
    def load(p): record(p); return readjson(p)
    batch = args.batch_root
    ds = args.dataset_root
    split_path = ds.parent / (ds.name + '_split.json')
    manual_path = ds.parent / (ds.name + '_manual_qc.json')
    split = load(split_path)
    manifest = load(ds / 'manifest.json')
    bm = load(batch / 'batch_manifest.json')
    qc = load(batch / 'rescan_qc.json')
    manual = load(manual_path)
    pairing_path = Path(bm['manifest'])
    pairing = load(pairing_path)
    exports = load(ds / 'export_report.json')
    validation = load(ds / 'validation_report.json')
    record(Path(__file__))
    evidence['git'] = {'head': git_read(args.pipeline_root, 'rev-parse', 'HEAD'), 'status': git_read(args.pipeline_root, 'status', '--short'), 'collection_history': git_read(args.pipeline_root, 'log', '-8', '--format=%H %aI %s', '--', 'scripts/collection/batch_collect_multitask.py', 'scripts/collection/prescreen_sources.py', 'scripts/collection/hrc_multitask_demo.py')}
    for name in ['scripts/collection/prescreen_sources.py', 'scripts/collection/batch_collect_multitask.py', 'scripts/collection/hrc_multitask_demo.py', 'configs/hrc_multitask_0001.yaml', 'scripts/evaluation/rescan_qc_full_phase.py', 'cmd_collect_endpt.sh']:
        record(args.pipeline_root / name)
    evidence['collection_code_identity'] = {'status': 'Partially Verified', 'reason': 'Actual per-episode CLI arguments are retained. Current source is hashed; the working hrc_multitask_demo.py is modified and no run-bound code/config snapshot hash is stored in final metadata. Current code is contextual evidence, not asserted identical to the July 22 collection code.'}
    evidence['split'] = split
    evidence['manual_qc'] = manual
    evidence['manifest_selection'] = {'batch_manifest': str(batch / 'batch_manifest.json'), 'pairing_manifest_referenced_by_final_batch': str(pairing_path), 'pairing_entries': len(pairing['episodes']), 'batch_entries': len(bm['episodes']), 'batch_non_episodes': {k: v for k, v in bm.items() if k != 'episodes'}, 'dataset_data_root_matches': Path(manifest['data_root']).resolve() == batch.resolve(), 'split_batch_root_matches': Path(split['batch_root']).resolve() == batch.resolve(), 'qc_sha256_matches_frozen_split': record(batch / 'rescan_qc.json')['sha256'] == split['qc_report_sha256'], 'batch_sha256_matches_frozen_split': record(batch / 'batch_manifest.json')['sha256'] == split['batch_manifest_sha256']}
    by_b = {e['idx']: e for e in bm['episodes']}
    by_p = {e['idx']: e for e in pairing['episodes']}
    by_q = {e['idx']: e for e in qc['episodes']}
    by_m = {e['source_episode_id']: e for e in manifest['sources']}
    sources = sorted(split['expected_clean'])
    split_sets = {s: set(ids) for s, ids in split['source_splits'].items()}
    evidence['global_checks'] = {'source_count': len(sources), 'split_counts': {s: len(ids) for s, ids in split_sets.items()}, 'expected_matches_manifest_ids': sources == sorted(by_m), 'expected_matches_batch_clean': sources == sorted(e['idx'] for e in bm['episodes'] if e['verdict'] == 'CLEAN'), 'expected_matches_rescan_clean': sources == sorted(e['idx'] for e in qc['episodes'] if e['verdict'] == 'CLEAN'), 'split_union_matches': set(sources) == set.union(*split_sets.values()), 'split_manifest_equal': split['source_splits'] == manifest['source_splits'], 'batch_all65_pairing_paths_equal': all(e['trash_hdf5'] == by_p[e['idx']]['trash_hdf5'] and e['radio_hdf5'] == by_p[e['idx']]['radio_hdf5'] for e in bm['episodes'])}
    overlaps = {f'{a}--{b}': sorted(split_sets[a] & split_sets[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    batchlog_path = args.pipeline_root / 'logs/batch_rerun_v1.log'
    record(batchlog_path)
    batchlog = batchlog_path.read_text().splitlines()
    starts = {}
    for lineno, line in enumerate(batchlog, 1):
        m = re.search(r'\[batch\] ep(\d+): START trash=(\S+) radio=(\S+) -> (\S+)', line)
        if m: starts[int(m[1])] = {'line': lineno, 'trash_basename': m[2], 'radio_basename': m[3], 'log_path': m[4]}
    evidence['batchlog_start_records'] = starts
    evidence['global_checks']['batchlog_all65_match'] = all(e['idx'] in starts and starts[e['idx']]['trash_basename'] == Path(e['trash_hdf5']).name and starts[e['idx']]['radio_basename'] == Path(e['radio_hdf5']).name and Path(starts[e['idx']]['log_path']).parent.parent.resolve() == batch.resolve() for e in bm['episodes'])
    lerobot = {}
    for s in ['train', 'val', 'test']:
        info = load(ds / s / 'meta/info.json')
        ep_rows = []
        columns = ['episode_index', 'length', 'stats/hrc.source_episode_id/min', 'stats/hrc.source_episode_id/max']
        for p in sorted((ds / s / 'meta/episodes').rglob('*.parquet')):
            record(p)
            ep_rows.extend(pq.read_table(p, columns=columns).to_pylist())
        observations = collections.defaultdict(lambda: {'sources': set(), 'rows': 0, 'raw_demo_counts': collections.Counter(), 'frame_min': None, 'frame_max': None, 'global_min': None, 'global_max': None})
        for p in sorted((ds / s / 'data').rglob('*.parquet')):
            record(p)
            t = pq.read_table(p, columns=['episode_index', 'hrc.source_episode_id', 'hrc.raw_demo_id', 'frame_index', 'hrc.raw_global_frame_index'])
            ep = np.asarray(t['episode_index']); sid = np.asarray(t['hrc.source_episode_id']); rid = np.asarray(t['hrc.raw_demo_id']); frame = np.asarray(t['frame_index']); glob = np.asarray(t['hrc.raw_global_frame_index'])
            for eid in np.unique(ep):
                mask = ep == eid; r = observations[int(eid)]; r['sources'].update(int(x) for x in np.unique(sid[mask])); r['rows'] += int(mask.sum()); r['raw_demo_counts'].update({int(k): int(v) for k, v in zip(*np.unique(rid[mask], return_counts=True))})
                for prefix, values in [('frame', frame[mask]), ('global', glob[mask])]:
                    lo, hi = int(values.min()), int(values.max()); r[prefix + '_min'] = lo if r[prefix + '_min'] is None else min(lo, r[prefix + '_min']); r[prefix + '_max'] = hi if r[prefix + '_max'] is None else max(hi, r[prefix + '_max'])
        observation_rows = []
        for ep, v in observations.items():
            v['sources'] = sorted(v['sources']); observation_rows.append({'episode_index': ep, **v})
        lerobot[s] = {'info_total_episodes': info['total_episodes'], 'info_total_frames': info['total_frames'], 'episode_metadata': ep_rows, 'actual_data_episode_columns': observation_rows, 'export_sources_match': exports['splits'][s]['sources'] == split['source_splits'][s]}
    evidence['lerobot'] = lerobot
    evidence['global_checks']['lerobot_actual_split_sources_equal'] = all(sorted({sid for r in data['actual_data_episode_columns'] for sid in r['sources']}) == sorted(split_sets[s]) for s, data in lerobot.items())
    evidence['global_checks']['lerobot_episode_and_row_totals_equal'] = all(len(data['actual_data_episode_columns']) == len(data['episode_metadata']) == data['info_total_episodes'] == len(split_sets[s]) and sum(r['rows'] for r in data['actual_data_episode_columns']) == data['info_total_frames'] == exports['splits'][s]['num_rows'] == validation['splits'][s]['num_rows'] for s, data in lerobot.items())
    actual_index = {}
    for s, data in lerobot.items():
        meta = {r['episode_index']: r for r in data['episode_metadata']}
        for row in data['actual_data_episode_columns']:
            for sid in row['sources']: actual_index.setdefault(sid, []).append({'split': s, **row, 'metadata': meta.get(row['episode_index'])})
    result_sources = []
    raw_rows = []
    for sid in sources:
        e, p, m, q = by_b[sid], by_p[sid], by_m[sid], by_q[sid]
        s = [k for k, v in split_sets.items() if sid in v]
        issues = []
        def check(name, ok):
            if not ok: issues.append(name)
            return bool(ok)
        ep = Path(m['hdf5_path']).parent
        commands_path = ep / 'annotations/commands.json'
        meta_path = ep / 'meta/episode_meta.json'
        log_path = batch / f'logs/ep_{sid:04d}.log'
        commands = load(commands_path)
        meta = load(meta_path)
        record(log_path)
        lines = log_path.read_text(errors='replace').splitlines()
        cli = None
        cli_line = None
        log_evidence = []
        for lineno, line in enumerate(lines, 1):
            if 'Passing the following args to the base kit application:' in line:
                cli = ast.literal_eval(line.split('application:', 1)[1].strip()); cli_line = lineno
            if any(x in line for x in ['[multitask] Base HDF5:', '[multitask] External replay:', '[multitask] CLI override: task_rounds[1].external_hdf5']):
                log_evidence.append({'line': lineno, 'text': line})
        def cli_arg(flag):
            return cli[cli.index(flag) + 1] if cli is not None and flag in cli else None
        cli_checks = {'raw_hdf5_equal': check('cli_raw_hdf5', cli_arg('--raw-hdf5') == e['trash_hdf5']), 'external_hdf5_equal': check('cli_external_hdf5', cli_arg('--external-hdf5') == '1:' + e['radio_hdf5']), 'output_dir_equal': check('cli_output_dir', cli_arg('--output-dir') == str(ep)), 'external_demo_0_logged': check('external_demo_0_log', any(f"External replay: {Path(e['radio_hdf5']).name} demo_0 [0,end)" in l for l in lines))}
        h5rec = record(m['hdf5_path'])
        with h5py.File(m['hdf5_path'], 'r') as f:
            data = f['data']; attrs = plain(dict(data.attrs)); env_args = json.loads(attrs['env_args']); fsm_states = json.loads(attrs['fsm_states']); execute_sid = fsm_states.index('ExecuteBehaviorTask')
            check('h5_raw_path', env_args.get('raw_hdf5') == e['trash_hdf5'])
            check('h5_num_demos', int(data.attrs['num_demos']) == 2)
            check('h5_total_frames', int(data.attrs['total_frames']) == m['num_frames'] == meta['num_frames'])
            rounds = []
            for r, key in [(0, 'trash_hdf5'), (1, 'radio_hdf5')]:
                g = data[f'demo_{r}']; ga = plain(dict(g.attrs)); actions = g['action'][:]; mask = g['obs/fsm_state_id'][:] == execute_sid
                rr = record(e[key]); rp = Path(e[key]);
                with h5py.File(rp, 'r') as rf:
                    rg = rf['data/demo_0']; raw_action = rg['action'][:]; raw_state = rg['state'][:]; raw_cfg = json.loads(rf['data'].attrs['config']); sf = json.loads(rf['data'].attrs['scene_file']);
                    semantic = {'action': arrinfo(raw_action), 'state': arrinfo(raw_state)}
                    scene = raw_cfg.get('scene', {}); task = raw_cfg.get('task', {}); inst = sf.get('metadata', {}).get('task', {}).get('inst_to_name', {})
                    raw_info = {**rr, 'source_id': sid, 'split': s[0], 'stage': r, 'upstream_demo': 'data/demo_0', 'semantic': semantic, 'scene_model': scene.get('scene_model'), 'scene_instance': scene.get('scene_instance'), 'activity_name': task.get('activity_name'), 'activity_definition_id': task.get('activity_definition_id'), 'activity_instance_id': task.get('activity_instance_id'), 'task_instance_mapping': inst, 'scene_versions': sf.get('versions'), 'num_raw_episodes': int(rf['data'].attrs.get('n_episodes', -1))}
                raw_info['inode_identity'] = [rr['device'], rr['inode']]
                raw_info['action_identity'] = semantic['action']['canonical_sha256']; raw_info['state_identity'] = semantic['state']['canonical_sha256']; raw_info['action_state_identity'] = raw_info['action_identity'] + ':' + raw_info['state_identity']
                raw_rows.append(raw_info)
                replay_action = actions[mask]
                eq = replay_action.shape == raw_action.shape and np.array_equal(replay_action, raw_action, equal_nan=True)
                check(f'round{r}_exact_replay_actions', eq)
                check(f'round{r}_pairing_raw_length', len(raw_action) == int(p['trash_len' if r == 0 else 'radio_len']))
                check(f'round{r}_length', len(actions) == m[f'demo{r}_length'] == int(ga['num_frames']))
                check(f'round{r}_success', int(ga['round_success']) == 1 and commands[r]['success'] is True and int(commands[r]['round_id']) == r)
                check(f'round{r}_command_frame', int(commands[r]['issued_frame']) == m[f'round{r}_issued_global'])
                rounds.append({'round': r, 'attrs': ga, 'raw_path': e[key], 'raw_identifier': str(rp.relative_to(rp.parent.parent)) + ':data/demo_0', 'hrc_frames': len(actions), 'raw_action_frames': len(raw_action), 'execute_frames': int(mask.sum()), 'exact_replay_action_match': eq, 'replay_actions': arrinfo(replay_action), 'raw_file_sha256': rr['sha256'], 'raw_action_sha256': semantic['action']['canonical_sha256'], 'raw_state_sha256': semantic['state']['canonical_sha256']})
        check('pairing_paths', e['trash_hdf5'] == p['trash_hdf5'] and e['radio_hdf5'] == p['radio_hdf5'])
        check('retained_clean', e['verdict'] == q['verdict'] == 'CLEAN')
        check('batch_output_path', e['out_dir'] == str(ep))
        check('source_split', s == [m['split']])
        lr = actual_index.get(sid, [])
        check('lerobot_source_single_episode_split', len(lr) == 1 and lr[0]['split'] == s[0])
        if len(lr) == 1:
            z = lr[0]; zm = z['metadata']
            check('lerobot_rows_frames', z['rows'] == zm['length'] == m['num_frames'])
            check('lerobot_metadata_source', zm['stats/hrc.source_episode_id/min'] == zm['stats/hrc.source_episode_id/max'] == [sid])
            check('lerobot_round_lengths', all(z['raw_demo_counts'].get(r) == m[f'demo{r}_length'] for r in [0, 1]))
            check('lerobot_global_frame_range', z['global_min'] == 0 and z['global_max'] == m['num_frames'] - 1)
        result_sources.append({'source_id': sid, 'split': s[0], 'hdf5_path': m['hdf5_path'], 'hdf5_sha256': h5rec['sha256'], 'trash_raw_path': e['trash_hdf5'], 'radio_raw_path': e['radio_hdf5'], 'pairing_manifest': str(pairing_path), 'pairing_entry_json_pointer': '/episodes/' + str(next(i for i, v in enumerate(pairing['episodes']) if v['idx'] == sid)), 'pairing_entry': p, 'batch_entry': e, 'rescan_qc_entry': q, 'qc_status': q['verdict'], 'manual_qc': manual.get('source_' + str(sid), 'not separately reviewed'), 'commands_path': str(commands_path), 'commands': commands, 'episode_meta_path': str(meta_path), 'episode_meta': meta, 'log_path': str(log_path), 'cli_line': cli_line, 'actual_cli_arguments': cli, 'log_evidence': log_evidence, 'cli_checks': cli_checks, 'hdf5_data_attrs': attrs, 'rounds': rounds, 'lerobot': lr, 'issues': issues, 'status': 'Verified' if not issues else 'Partially Verified'})
        print(f'source={sid:04d} split={s[0]} exact_action_matches={[x["exact_replay_action_match"] for x in rounds]} issues={issues}', flush=True)
    for stage in [0, 1]:
        rr = [r for r in raw_rows if r['stage'] == stage]
        print(f'round={stage} raw_count={len(rr)} bytes={sum(r["size_bytes"] for r in rr)} unique_file_sha256={len({r["sha256"] for r in rr})}', flush=True)
    evidence['sources'] = result_sources
    evidence['raw_files'] = raw_rows
    evidence['fixed_distribution'] = {str(stage): {key: dict(collections.Counter(str(r[key]) for r in raw_rows if r['stage'] == stage)) for key in ['scene_model', 'scene_instance', 'activity_name', 'activity_definition_id', 'activity_instance_id']} for stage in [0, 1]}
    evidence['run']['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    duplicate_checks = {k: group_dups(raw_rows, k) for k in ['path', 'realpath', 'inode_identity', 'sha256', 'action_identity', 'state_identity', 'action_state_identity']}
    cross_split = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        pair = {}
        for stage in [0, 1]:
            rr = [r for r in raw_rows if r['stage'] == stage]
            pair[str(stage)] = {key: sorted({str(r[key]) for r in rr if r['split'] == a} & {str(r[key]) for r in rr if r['split'] == b}) for key in ['path', 'realpath', 'inode_identity', 'sha256', 'action_identity', 'state_identity', 'action_state_identity']}
        cross_split[a + '--' + b] = pair
    evidence['checks'] = {'A_source_id_overlap': overlaps, 'B_trash_reuse': {k: [g for g in v if any(m['stage'] == 0 for m in g['members'])] for k, v in duplicate_checks.items()}, 'C_radio_reuse': {k: [g for g in v if any(m['stage'] == 1 for m in g['members'])] for k, v in duplicate_checks.items()}, 'D_cross_split_reuse': cross_split, 'E_all90_duplicate_groups': duplicate_checks, 'F_unverifiable_sources': [{'source_id': s['source_id'], 'issues': s['issues']} for s in result_sources if s['issues']]}
    evidence['summary'] = {'retained_sources': len(result_sources), 'source_split_counts': {s: len(v) for s, v in split_sets.items()}, 'verified_sources': sum(not s['issues'] for s in result_sources), 'unverifiable_sources': [s['source_id'] for s in result_sources if s['issues']], 'raw_files': len(raw_rows), 'raw_bytes_hashed': sum(r['size_bytes'] for r in raw_rows), 'unique_raw_realpaths': len({r['realpath'] for r in raw_rows}), 'unique_raw_inodes': len({str(r['inode_identity']) for r in raw_rows}), 'unique_raw_file_hashes': len({r['sha256'] for r in raw_rows}), 'unique_raw_action_hashes': len({r['action_identity'] for r in raw_rows}), 'unique_raw_state_hashes': len({r['state_identity'] for r in raw_rows}), 'exact_action_matches': sum(r['exact_replay_action_match'] for s in result_sources for r in s['rounds']), 'cross_split_reuse_groups': sum(bool(ids) for p in cross_split.values() for r in p.values() for ids in r.values()), 'elapsed_seconds': round(time.time() - started, 3)}
    fields = ['hrc_source_id', 'split', 'final_hdf5_path', 'final_hdf5_sha256', 'round0_raw_path', 'round0_raw_identifier', 'round0_raw_sha256', 'round0_action_sha256', 'round0_state_sha256', 'round0_replay_exact_actions', 'round1_raw_path', 'round1_raw_identifier', 'round1_raw_sha256', 'round1_action_sha256', 'round1_state_sha256', 'round1_replay_exact_actions', 'pairing_manifest_path', 'pairing_entry_json_pointer', 'pairing_entry_json', 'batch_manifest_path', 'qc_status', 'manual_qc', 'commands_path', 'episode_meta_path', 'actual_execution_log_path', 'actual_cli_line', 'lerobot_episode_index', 'status', 'unverifiable_items']
    csv_path = out / 'reports/UPSTREAM_SOURCE_MAPPING.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for s in result_sources:
            r0, r1 = s['rounds']
            w.writerow({'hrc_source_id': s['source_id'], 'split': s['split'], 'final_hdf5_path': s['hdf5_path'], 'final_hdf5_sha256': s['hdf5_sha256'], 'round0_raw_path': s['trash_raw_path'], 'round0_raw_identifier': r0['raw_identifier'], 'round0_raw_sha256': r0['raw_file_sha256'], 'round0_action_sha256': r0['raw_action_sha256'], 'round0_state_sha256': r0['raw_state_sha256'], 'round0_replay_exact_actions': r0['exact_replay_action_match'], 'round1_raw_path': s['radio_raw_path'], 'round1_raw_identifier': r1['raw_identifier'], 'round1_raw_sha256': r1['raw_file_sha256'], 'round1_action_sha256': r1['raw_action_sha256'], 'round1_state_sha256': r1['raw_state_sha256'], 'round1_replay_exact_actions': r1['exact_replay_action_match'], 'pairing_manifest_path': s['pairing_manifest'], 'pairing_entry_json_pointer': s['pairing_entry_json_pointer'], 'pairing_entry_json': json.dumps(s['pairing_entry'], ensure_ascii=False), 'batch_manifest_path': str(batch / 'batch_manifest.json'), 'qc_status': s['qc_status'], 'manual_qc': s['manual_qc'], 'commands_path': s['commands_path'], 'episode_meta_path': s['episode_meta_path'], 'actual_execution_log_path': s['log_path'], 'actual_cli_line': s['cli_line'], 'lerobot_episode_index': s['lerobot'][0]['episode_index'] if len(s['lerobot']) == 1 else '', 'status': s['status'], 'unverifiable_items': ';'.join(s['issues'])})
    evidence['outputs'] = {'source_mapping': {'path': str(csv_path), 'sha256': sha(csv_path)}}
    ep = out / 'evidence/provenance_verification.json'
    ep.write_text(json.dumps(plain(evidence), ensure_ascii=False, indent=2) + '\n')
    hp = out / 'evidence/provenance_hash_manifest.json'
    hp.write_text(json.dumps({'run': evidence['run'], 'files': evidence['files'], 'generated': [{'path': str(ep), 'sha256': sha(ep)}, {'path': str(csv_path), 'sha256': sha(csv_path)}]}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(evidence['summary'], ensure_ascii=False, indent=2))
    print('EVIDENCE', ep, 'SHA256', sha(ep))
    print('HASH_MANIFEST', hp, 'SHA256', sha(hp))


if __name__ == '__main__': main()
