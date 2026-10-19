#!/usr/bin/env python3
"""Rebuild all 18 native dissertation tables from frozen records and original text.

Usage: python build_tables.py --data-root /root/autodl-tmp
Numeric audit cells are formatted only after reading frozen aggregates. No OCR,
model inference, training or changes to research inputs are performed.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
MODELS = ('A20', 'B20', 'C20')
EXPECTED = {'3.1', '3.2', '4.1', '4.2', '5.1', '5.2', '5.3', '5.4', '5.5',
            'A.1', 'A.2', 'A.3', 'A.4', 'A.5', 'A.6', 'A.7', 'A.8', 'A.9'}
ORIGINAL = '/root/NTU_Dissertation_Reviewed_v6_editable.docx'


def load(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def build_tables(data_root):
    outputs = data_root / 'openpi_hrc/outputs'
    audit = outputs / 'dissertation_final_audit'
    probe_path = outputs / 'dissertation_conditioning_probe/summary.json'
    visibility = outputs / 'dissertation_target_visibility/summary.csv'
    manifest = load(audit / 'anchor_manifest.json')
    probe_manifest = load(outputs / 'dissertation_conditioning_probe/anchor_manifest.json')
    dataset_manifest_path = data_root / 'lerobot_datasets/hrc_subinstr_rerun_v1/manifest.json'
    dataset_manifest = load(dataset_manifest_path)
    annotation_path = data_root / 'lerobot_datasets/hrc_subinstructions_rerun_v1.json'
    annotations = load(annotation_path)['sources']
    comparison = read_csv(audit / 'comparison.csv')
    lookup = {(r['model'], r['context'], r['action_group'], r['metric']): r for r in comparison}
    assert len(lookup) == len(comparison), 'Duplicate frozen comparison metric keys'
    probe = load(probe_path)['models']
    split_path = ROOT / 'tables/split_summary.csv'
    splits = read_csv(split_path)
    ids = [str(s['source_episode_id']) for s in dataset_manifest['sources']]
    assert len(ids) == len(set(ids)) == 45
    assert sum(int(s['sources']) for s in splits) == len(ids)
    assert sum(int(s['frames']) for s in splits) == sum(s['num_frames'] for s in dataset_manifest['sources'])
    assert manifest['num_anchors'] == len(manifest['anchors']) == 652
    assert Counter(a['context'] for a in manifest['anchors']) == manifest['selected_counts']
    assert manifest['flow_seeds'] == [0, 1, 2, 3, 4]
    tables = {}

    def add(no, caption, headers, rows, source):
        rows = [[str(v) for v in row] for row in rows]
        assert all(len(row) == len(headers) for row in rows), no
        assert no not in tables
        tables[no] = dict(caption=caption, headers=headers, rows=rows, source=str(source))

    def stat(model, context, group, metric, digits=4, spread=True):
        r = lookup[model, context, group, metric]
        assert r['mean'] != '', (model, context, group, metric)
        value = f"{float(r['mean']):.{digits}f}"
        if spread:
            assert int(r['n']) == 5, 'Expected five inference-flow repetitions'
            value += f" ± {float(r['std']):.{digits}f}"
        return value

    def pstat(model, context, kind):
        rows = [r for r in probe[model]['by_context'] if r['split'] == 'val'
                and r['context'] == context and r['metric'] == f'arms_{kind}_over_noise']
        assert len(rows) == 1
        return rows[0]

    def interval(r):
        return f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"

    add('3.1', 'Controller scope during dataset generation and policy evaluation.',
        ['Phase or function', 'Dataset generation', 'Policy evaluation'], [
            ['Human-facing and transition phases', 'Scripted FSM', 'Scripted FSM'],
            ['ExecuteBehaviorTask', 'Replayed BEHAVIOR state–action trajectory', 'π0.5 policy']],
        ORIGINAL + '; Table 3.1, original image4.png; faithful text transcription')
    uses = {'train': 'Policy training and baseline fitting', 'val': 'Main diagnostic evaluation',
            'test': 'Held out from training, selection, and policy performance'}
    labels = {'train': 'Training', 'val': 'Validation', 'test': 'Test'}
    split_rows = [[labels[r['split']], r['sources'], f"{int(r['frames']):,}", uses[r['split']]] for r in splits]
    split_rows.append(['Total', len(ids), f"{sum(int(r['frames']) for r in splits):,}", '—'])
    add('3.2', 'Source-level dataset split.', ['Split', 'Sources', 'Frames', 'Use'], split_rows, split_path)
    add('4.1', 'Main policy configurations. A20 to B20 combines prompt and sampling changes. '
        'B20 to C20 changes the normalisation pipeline for both state and action.',
        ['Setting', *MODELS], [
            ['Instruction', 'Round-level', 'Local sub-instructions', 'Local sub-instructions'],
            ['Anchor sampling', 'Natural', '80% ordinary, 20% decision target', 'Same as B20'],
            ['State scaling', 'z-score', 'z-score', 'Quantile'],
            ['Action scaling', 'z-score', 'z-score', 'Quantile'],
            ['Training updates', '20,000', '20,000', '20,000'],
            ['Training seed', '42', '42', '42'], ['Visual tower', 'Frozen', 'Frozen', 'Frozen']],
        ORIGINAL + '; Table 4.1, original image6.png; configuration text, not measured outcomes')
    pcounts = Counter(a['split'] for a in probe_manifest['anchors'])
    add('4.2', 'Populations and anchor rules for the offline protocols. The conditioning coverage subset '
        'and all-source geometry analysis are reported separately from validation policy performance.',
        ['Setting', 'Teacher-forced audit', 'Conditioning probe', 'Frustum audit'], [
            ['Population', f"{manifest['num_anchors']} validation anchors",
             f"{pcounts['val']} validation; {pcounts['train_coverage']} train-coverage anchors", '45 sources across all splits'],
            ['Policy checkpoints', 'A20, B20, C20', 'B20, C20', 'No policy inference'],
            ['Local prompt stability', 'Not required within a stable coarse context', 'Required through the horizon', 'Separate geometric populations'],
            ['Flow seeds', ', '.join(map(str, manifest['flow_seeds'])), '0 reference; 1 noise perturbation', 'Not applicable'],
            ['Purpose', 'Matched offline policy comparison', 'Local output sensitivity', 'Descriptive target geometry']],
        f'{audit}/anchor_manifest.json; {outputs}/dissertation_conditioning_probe/anchor_manifest.json; '
        f'{ORIGINAL}, Table 4.2 for protocol descriptions')
    grouped = defaultdict(lambda: [0, 0])
    for r in read_csv(visibility):
        if (r['split'], r['context'], r['population']) == ('val', 'pickup', 'valid_anchors'):
            assert r['position'] in ('early', 'middle', 'late')
            grouped[r['segment']][0] += int(r['total'])
            grouped[r['segment']][1] += int(r['target_center_in_frustum'])
    segment_names = [('Walk to / acquire trash can', 'Walk to the trash can and pick it up'),
                     ('Pick up soda can', 'Pick up the soda can'),
                     ('Deposit can', 'Drop the can into the trash can you are holding'),
                     ('Put trash can down', 'Put the trash can down')]
    assert set(grouped) == {key for _, key in segment_names}
    visibility_rows = [[label, f'{grouped[key][0]:,}', f'{grouped[key][1]:,}',
                        f'{grouped[key][1] / grouped[key][0]:.4f}'] for label, key in segment_names]
    assert sum(v[0] for v in grouped.values()) == 12302 and sum(v[1] for v in grouped.values()) == 10911
    add('5.1', 'Validation pickup target-centre geometric frustum coverage by task segment. '
        'The metric does not model occlusion or rendered-pixel visibility.',
        ['Segment', 'Valid anchors', 'In frustum', 'Ratio'], visibility_rows,
        f'{visibility}; val/pickup/valid_anchors; sum counts across early/middle/late before division')
    specs = [('Pickup arm skill', 'pickup', 'arms', 'skill'), ('Pickup motion ratio', 'pickup', 'arms', 'arm_motion_ratio'),
             ('Pickup base skill', 'pickup', 'base', 'skill'), ('Radio arm skill', 'radio', 'arms', 'skill'),
             ('Follow base skill', 'follow', 'base', 'skill')]
    comparison_source = str(audit / 'comparison.csv')
    add('5.2', 'Unified validation audit. Values are mean ± sample standard deviation over five inference '
        'flow-noise seeds; each model is one trained checkpoint.', ['Metric', *MODELS],
        [[label, *[stat(m, context, group, metric) for m in MODELS]] for label, context, group, metric in specs], comparison_source)
    event_rows = []
    for context in ('pickup', 'radio'):
        for model in MODELS:
            assert int(float(lookup[model, context, 'grippers', 'gripper_expert_events']['mean'])) == (14 if context == 'pickup' else 9)
            event_rows.append([context.capitalize(), model, *[stat(model, context, 'grippers', f'gripper_{k}') for k in ('precision', 'recall', 'f1')]])
    add('5.3', 'Gripper event precision, recall and F1. Values are mean ± sample standard deviation over five '
        'inference flow-noise seeds. Expert transitions are pooled over sampled chunks: 14 pickup and 9 radio transitions per seed.',
        ['Context', 'Model', 'Precision', 'Recall', 'F1'], event_rows, comparison_source)
    baseline_keys = ('lookup_A', 'lookup_A_copycat_base', 'lookup_B', 'lookup_B_copycat_base')
    baseline_names = ('Lookup A', 'Lookup A + persistence', 'Lookup B', 'Lookup B + persistence')
    groups = [('pickup', 'arms'), ('pickup', 'base'), ('radio', 'arms'), ('radio', 'base'), ('follow', 'base')]
    add('5.4', 'Shared-manifest trivial baselines. Lookup is fitted on training anchors only. The persistence '
        'variant repeats the previous base command. Lookup B is also the prompt-matched reference for C20.',
        ['Baseline', 'Pickup arms', 'Pickup base', 'Radio arms', 'Radio base', 'Follow base'],
        [[name, *[stat(key, c, g, 'skill', spread=False) for c, g in groups]] for key, name in zip(baseline_keys, baseline_names)], comparison_source)
    order = [('B20', 'pickup'), ('C20', 'pickup'), ('B20', 'radio'), ('C20', 'radio')]
    add('5.5', 'Validation conditioning ratios for arm outputs. Brackets show 95% anchor-level percentile-bootstrap '
        'intervals for the mean. Anchors, not source trajectories, are resampled.',
        ['Model / context', 'Image / noise', 'State / noise', 'n'],
        [[f'{m} {c}', *[f"{pstat(m, c, k)['mean']:.4f} {interval(pstat(m, c, k))}" for k in ('image', 'state')],
          pstat(m, c, 'image')['n']] for m, c in order], f'{probe_path}; split=val; per-anchor ratios')
    duration_labels = [['Walk to / acquire trash can', 'Pick up soda can 1', 'Deposit can 1', 'Pick up soda can 2',
                        'Deposit can 2', 'Pick up soda can 3', 'Deposit can 3', 'Put trash can down', 'Stand by'],
                       ['Walk to / acquire trash can', 'Turn on radio receiver', 'Put trash can down', 'Stand by']]
    for rnd in (0, 1):
        rows = []
        for index, label in enumerate(duration_labels[rnd] + [f'Round {rnd} total']):
            values = []
            for sid in ids:
                record = annotations[sid][str(rnd)]
                start, stop = ((record['segments'][index]['start'], record['segments'][index]['stop'])
                               if index < len(duration_labels[rnd]) else (record['execute_start'], record['execute_stop']))
                values.append(stop - start)
            values.sort()
            rows.append([label, values[0], int(median(values)), values[math.ceil(.90 * (len(values) - 1))], values[-1]])
        add(f'A.{rnd+1}', f'Round {rnd} expert segment durations. Frames at 30 Hz across the 45 retained sources.',
            ['Segment', 'Min', 'Median', '90th percentile', 'Max'], rows,
            f'{annotation_path}; source IDs from {dataset_manifest_path}; stop-start, total=execute_stop-execute_start; '
            'p90 uses higher order statistic ceil(0.90*(n-1)), matching the original integer table')
    parameters = [
        ['Base model', 'π0.5 flow-matching VLA'], ['Physical state/action dimensions', '23'], ['Model action dimensions', '32'],
        ['Action horizon', str(manifest['action_horizon'])], ['Head-image model input', '224 × 224, resized with padding'],
        ['Combined prompt/state token budget', '200 tokens'], ['State input', 'Discrete-state input enabled'],
        ['Fine-tuning', 'LoRA variants with frozen visual tower'], ['EMA', 'Disabled'], ['Training seed', '42; one run per configuration'],
        ['Optimiser', 'AdamW'], ['AdamW β1, β2', '0.9, 0.95'], ['AdamW ε', '10⁻⁸'], ['Weight decay', '10⁻¹⁰'],
        ['Global gradient-norm clipping', '1.0'], ['Warm-up', '1,000 updates'], ['Peak learning rate', '2.5 × 10⁻⁵'],
        ['Cosine schedule length', '30,000 updates, including warm-up'], ['End-of-schedule learning rate', '2.5 × 10⁻⁶'],
        ['Physical microbatch size', '2'], ['Gradient accumulation', '4 microbatches per update'], ['Effective batch size', '8'],
        ['Main training length', '20,000 optimiser updates'], ['Flow-loss dimensions in A/B/C', 'All 32, including padding'],
        ['Training-time validation', 'Every 1,000 updates; 10 × 2 examples'], ['Checkpoint save / retention intervals', '1,000 / 5,000 updates'],
        ['Offline action integration', '10 Euler steps'], ['Main audit flow-noise seeds', ', '.join(map(str, manifest['flow_seeds']))],
        ['Conditioning reference / alternate seeds', f"{probe_manifest['reference_flow_seed']} / {probe_manifest['alternate_flow_seed']}"],
        ['Protocol seed', str(manifest['protocol_seed'])], ['Conditioning bootstrap resamples', '10,000']]
    add('A.3', 'Shared hyperparameters for the main 20k runs.', ['Parameter', 'Value'], parameters,
        f'{ORIGINAL}; original paragraphs 796–802; audit/probe manifests for protocol fields; no OCR inference')
    configs = [load(audit / m / 'summary.json') for m in MODELS]
    assert all(Path(s['checkpoint']).name == '20000' for s in configs)
    add('A.4', 'Checkpoint configuration identifiers. Each experiment uses the checkpoint directory named 20000.',
        ['Model', 'Configuration', 'Experiment'], [[m, s['config'], Path(s['checkpoint']).parent.name] for m, s in zip(MODELS, configs)],
        f'{audit}/{{A20,B20,C20}}/summary.json; config and checkpoint fields')
    add('A.5', 'Frozen teacher-forced validation manifest.', ['Context', 'Candidate anchors', 'Selected anchors'],
        [[c.capitalize(), f"{manifest['candidate_counts'][c]:,}", manifest['selected_counts'][c]] for c in ('pickup', 'follow', 'radio', 'wait')]
        + [['Total', '—', manifest['num_anchors']]], audit / 'anchor_manifest.json')
    add('A.6', 'Raw arm MAE across all four contexts. Model values are mean ± sample standard deviation over five inference flow-noise seeds.',
        ['Context', 'No-motion', *MODELS], [[c.capitalize(), stat('A20', c, 'arms', 'null_mae', 6, False),
          *[stat(m, c, 'arms', 'raw_mae', 6) for m in MODELS]] for c in ('wait', 'pickup', 'follow', 'radio')], comparison_source)
    add('A.7', 'Predicted and matched gripper transitions. Values are mean ± sample standard deviation over five '
        'inference flow-noise seeds. Expert counts are 14 for pickup and 9 for radio on every seed.',
        ['Model', 'Context', 'Predicted', 'True positives'], [[m, c.capitalize(),
          stat(m, c, 'grippers', 'gripper_predicted_events', 1), stat(m, c, 'grippers', 'gripper_true_positives', 1)]
          for c in ('pickup', 'radio') for m in MODELS], comparison_source)
    add('A.8', 'Full validation conditioning summary for arm outputs. Intervals are 95% anchor-level percentile-bootstrap intervals.',
        ['Model / context', 'Metric', 'Mean', 'Median', 'Std.', '95% anchor BI', 'n'],
        [[f'{m} {c}', f'{k}/noise', f"{pstat(m,c,k)['mean']:.4f}", f"{pstat(m,c,k)['median']:.4f}",
          f"{pstat(m,c,k)['std']:.4f}", interval(pstat(m,c,k)), pstat(m,c,k)['n']]
          for m, c in order for k in ('image', 'state')], f'{probe_path}; split=val; arms_*_over_noise')
    add('A.9', 'Additional validation pickup image-perturbation ratios.',
        ['Model', 'Horizontal flip / noise', 'Black / noise', 'White / noise', 'Random / noise'],
        [[m, *[f"{pstat(m,'pickup',k)['mean']:.4f}" for k in ('hflip', 'black', 'white', 'random_image')]] for m in ('B20', 'C20')],
        f'{probe_path}; split=val, context=pickup; means of per-anchor ratios')
    assert set(tables) == EXPECTED and len(tables) == 18
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp'))
    parser.add_argument('--output', type=Path, default=ROOT / 'tables/native_tables.json')
    args = parser.parse_args()
    result = build_tables(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'Wrote {len(result)} tables to {args.output}')
