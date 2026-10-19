#!/usr/bin/env python3
"""Local-only research inventory and optional source snapshot. No network/git/imports.

Default: write inventories. --snapshot: additionally copy eligible small sources,
licenses, and a filesystem diff against the explicitly configured OpenPI baseline.
Never follow symlinks, read credential files, hash large data, or delete anything.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import os
from pathlib import Path
import re
import stat
from zoneinfo import ZoneInfo

STAGING = Path(__file__).resolve().parents[1]
PRIVATE = Path('/root/dissertation_local_audit')
ROOTS = {
    'hrc_pipeline': Path('/root/hrc_pipeline'),
    'hrc_pipeline_traintest': Path('/root/autodl-tmp/hrc_pipeline_traintest'),
    'openpi_hrc': Path('/root/autodl-tmp/openpi_hrc'),
}
BASELINE = Path('/root/autodl-tmp/openpi')
LIMIT = 1024 * 1024
PRUNE = {'.git', '.venv', 'venv', 'env', 'envs', '__pycache__', '.pytest_cache',
         '.ruff_cache', '.mypy_cache', 'node_modules', '.cache', 'cache', 'caches',
         'hf_cache', 'wandb', 'appdata', 'kit_cache', 'pip_cache', '.vscode',
         'tmp_notion_pages', '.aws', '.ssh', '.config', '.codex'}
THIRD = {'OmniGibson', 'habitat-lab', 'NaVILA', 'third_party'}
CODE_EXT = {'.py', '.sh', '.yaml', '.yml', '.json', '.toml', '.patch', '.ini', '.cfg'}
MODEL_EXT = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.tflite', '.bin'}
DATA_EXT = {'.hdf5', '.h5', '.npz', '.npy', '.pkl', '.pickle', '.parquet', '.arrow'}
ASSET_EXT = {'.usd', '.usda', '.usdc', '.glb', '.gltf', '.fbx', '.obj', '.stl',
             '.jpg', '.jpeg', '.png', '.exr', '.hdr', '.urdf', '.mp4', '.mov'}
SENSITIVE_NAMES = {'credentials', 'credentials.json', 'token.json', 'tokens.json',
                   'id_rsa', 'id_ed25519', '.netrc', '.npmrc', '.pypirc'}
SECRET_PATTERNS = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{25,}\b'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(r'\bhf_[A-Za-z0-9]{25,}\b'),
    re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b'),
    re.compile(r'''(?im)^\s*(?:export\s+)?(?:[A-Z_]*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET_ACCESS_KEY|PASSWORD))\s*[:=]\s*["']?[A-Za-z0-9_+/.=-]{16,}["']?\s*$'''),
    re.compile(r'https?://[^\s/:@]+:[^\s/@]{8,}@'),
]
FIELDS = ['root_alias', 'relative_path', 'category', 'entry_kind', 'bytes',
          'mtime_utc', 'sha256', 'hash_policy', 'snapshot_state', 'license_gate', 'snapshot_sha256', 'sanitization_policy']
CATEGORIES = {
    'A': '项目研究代码/配置候选；分类不证明权属',
    'B': 'OpenPI相对本地基线的改动/新增适配代码',
    'C': '第三方软件或未改动OpenPI上游代码',
    'D': '输入数据/模型资产/转换派生素材；不复制',
    'E': '模型权重/checkpoint；不复制不hash',
    'F': '运行输出/日志/结果/归档；不复制',
    'G': '环境、缓存、VCS、编辑器等排除目录/文件',
    'H': '敏感或私有材料/内部文档/未分类材料；不复制',
}


def sensitive_name(path: Path) -> bool:
    name = path.name.lower()
    return name in SENSITIVE_NAMES or name == '.env' or name.startswith('.env.') or name.endswith(('.pem', '.key', '.p12', '.pfx'))


def license_file(path: Path) -> bool:
    return path.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING'))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(alias: str, rel: Path) -> bool:
    """Conservative source selection; OpenPI includes differences, never full tree."""
    if any(part in PRUNE | THIRD for part in rel.parts) or sensitive_name(rel):
        return False
    if len(rel.parts) == 1 and license_file(rel):
        return True
    if alias == 'openpi_hrc':
        eligible = rel.parts[0] in {'scripts', 'src', 'packages'} and rel.suffix in CODE_EXT
        eligible = eligible or rel.as_posix() in {'pyproject.toml', '.python-version'}
        if not eligible:
            return False
        p = ROOTS[alias] / rel
        q = BASELINE / rel
        return not q.is_file() or p.read_bytes() != q.read_bytes()
    if rel.parts[0] in {'scripts', 'configs', 'patches'}:
        return rel.suffix in CODE_EXT
    return len(rel.parts) == 1 and rel.suffix == '.sh' and rel.name.startswith(('cmd', 'setup_', 'submit'))


def classify(alias: str, rel: Path, chosen: bool) -> str:
    parts = set(rel.parts)
    low = rel.as_posix().lower()
    if sensitive_name(rel):
        return 'H'
    if parts & PRUNE:
        return 'G'
    if parts & THIRD:
        return 'C'
    if parts & {'checkpoints', 'weights', 'models', 'act_runs'} or rel.suffix.lower() in MODEL_EXT:
        # src/openpi/models is source, not a weight directory.
        if not (rel.parts[0] == 'src' and rel.suffix in CODE_EXT):
            return 'E'
    if parts & {'outputs', 'logs', 'videos', '_compare_fix3'} or rel.suffix.lower() in {'.log', '.tar', '.gz', '.zip'}:
        return 'F'
    if parts & {'assets', 'datasets', 'data', 'textures'} or rel.suffix.lower() in DATA_EXT | ASSET_EXT:
        return 'D'
    if alias == 'openpi_hrc':
        return 'B' if chosen and not license_file(rel) else 'C'
    if rel.suffix in CODE_EXT or license_file(rel) or (len(rel.parts) == 1 and rel.name == 'README.md'):
        return 'A'
    return 'H'


def walk(root: Path):
    """Emit every non-pruned file, plus explicit records for pruned subtrees."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda item: item.name)
        except OSError as exc:
            yield current, 'unreadable_directory', type(exc).__name__
            continue
        for entry in entries:
            p = Path(entry.path)
            if entry.is_symlink():
                yield p, 'symlink_not_followed', ''
            elif entry.is_dir(follow_symlinks=False):
                if entry.name in PRUNE:
                    yield p, 'directory_skipped', 'pruned environment/cache/private subtree'
                else:
                    stack.append(p)
            elif entry.is_file(follow_symlinks=False):
                yield p, 'file', ''


def scan_text(path: Path) -> tuple[bytes, bool]:
    content = path.read_bytes()
    text = content.decode('utf-8')  # undecodable candidates must not be copied
    return content, any(pattern.search(text) for pattern in SECRET_PATTERNS)


def sanitize_historical_paths(content: bytes) -> tuple[bytes, int]:
    text = content.decode('utf-8')
    # Generic historical workstation paths; do not encode personal identifiers.
    pattern = re.compile(r"/media/[^/\s\"']+/home/[^/\s\"']+/[^/\s\"']+")
    text, count = pattern.subn('/path/to/external_project', text)
    return text.encode('utf-8'), count


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', action='store_true', help='Copy eligible source files locally; never publish.')
    args = parser.parse_args()
    rows = []
    secret_paths = []
    copies = Counter()
    patch_parts = []
    baseline_rows = []
    now = datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
    for alias, root in ROOTS.items():
        if not root.is_dir():
            raise SystemExit(f'Required research root missing: {alias}')
        for path, kind, note in walk(root):
            rel = path.relative_to(root)
            st = path.lstat()
            is_regular = kind == 'file' and stat.S_ISREG(st.st_mode)
            is_small = is_regular and st.st_size <= LIMIT
            chosen = is_small and selected(alias, rel)
            category = classify(alias, rel, chosen)
            row = dict(root_alias=alias, relative_path=rel.as_posix(), category=category,
                       entry_kind=kind, bytes=st.st_size,
                       mtime_utc=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                       sha256='', hash_policy='metadata_only', snapshot_state='excluded',
                       license_gate='NOT_SELECTED_FOR_PUBLICATION', snapshot_sha256='', sanitization_policy='none')
            if chosen:
                row['license_gate'] = ('AUTHORIZED_CUSTOM_CHANGES_PRESERVE_UPSTREAM_TERMS' if alias == 'openpi_hrc'
                                       else 'AUTHORIZED_THESIS_AND_DESIGNATED_REPOSITORY_ORIGINAL_CODE')
            elif category in {'C', 'D', 'E'}:
                row['license_gate'] = 'THIRD_PARTY_OR_DATA_EXCLUDED'
            if note:
                row['snapshot_state'] = note
            if not is_small:
                row['hash_policy'] = 'not_hashed_large_or_nonregular'
            elif category in {'A', 'B', 'C'} and (rel.suffix in CODE_EXT or license_file(rel)):
                row['sha256'] = digest(path)
                row['hash_policy'] = 'sha256_small_source_only'
            if chosen:
                try:
                    content, has_secret = scan_text(path)
                except (UnicodeError, OSError):
                    row['snapshot_state'] = 'excluded_unreadable_or_nontext'
                    rows.append(row)
                    continue
                if has_secret:
                    secret_paths.append(f'{alias}/{rel.as_posix()}')
                    row['snapshot_state'] = 'excluded_secret_pattern'
                    row['license_gate'] = 'EXCLUDED_SECRET_PATTERN'
                    row['category'] = 'H'
                    row['sha256'] = ''
                    row['hash_policy'] = 'secret_candidate_no_digest_exported'
                else:
                    content, masked_count = sanitize_historical_paths(content)
                    row['snapshot_sha256'] = hashlib.sha256(content).hexdigest()
                    row['sanitization_policy'] = f'historical_private_path_prefix_masked:{masked_count}' if masked_count else 'none'
                    row['snapshot_state'] = 'eligible_authorized_repo_candidate'
                    if args.snapshot:
                        dest = STAGING / 'source_snapshots' / alias / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(content)
                        os.chmod(dest, 0o644)
                        row['snapshot_state'] = 'copied_authorized_repo_candidate'
                        copies[alias] += 1
                    if alias == 'openpi_hrc' and not license_file(rel):
                        before = BASELINE / rel
                        exists = before.is_file()
                        original = before.read_text(encoding='utf-8') if exists else ''
                        baseline_rows.append({'relative_path': rel.as_posix(), 'baseline_exists': str(exists),
                                              'baseline_sha256': digest(before) if exists else '',
                                              'snapshot_sha256': hashlib.sha256(content).hexdigest()})
                        patch_parts.append(f'diff --git a/{rel.as_posix()} b/{rel.as_posix()}\n')
                        for diff_line in difflib.unified_diff(original.splitlines(keepends=True),
                            content.decode('utf-8').splitlines(keepends=True),
                            fromfile=f'a/{rel.as_posix()}' if exists else '/dev/null',
                            tofile=f'b/{rel.as_posix()}'):
                            patch_parts.append(diff_line)
                            if not diff_line.endswith('\n'):
                                patch_parts.append('\n\\ No newline at end of file\n')
            rows.append(row)
    rows.sort(key=lambda r: (r['root_alias'], r['relative_path']))
    report = STAGING / 'reports'
    report.mkdir(parents=True, exist_ok=True)
    write_csv(report / 'FILE_CLASSIFICATION.csv', rows, FIELDS)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    os.chmod(PRIVATE, 0o700)
    private_rows = [{**r, 'absolute_path': str(ROOTS[r['root_alias']] / r['relative_path'])} for r in rows]
    private_file = PRIVATE / 'LOCAL_RESOURCE_MANIFEST.csv'
    write_csv(private_file, private_rows, FIELDS + ['absolute_path'])
    os.chmod(private_file, 0o600)
    by_root = defaultdict(Counter)
    by_bytes = defaultdict(Counter)
    for r in rows:
        by_root[r['root_alias']][r['category']] += 1
        if r['entry_kind'] == 'file':
            by_bytes[r['root_alias']][r['category']] += r['bytes']
    lines = ['# 研究文件清单与分类', '', f'生成时间：{now}', '',
        '本清单仅覆盖三个明确研究root；发布候选版只含root_alias/relative_path，私有绝对路径清单独立保管。',
        '权利人已明确授权本论文及指定仓库使用原创代码；保留原LICENSE、不重许可。第三方资产/权重/原始数据仍排除，代码待最终安全审查后可上传。分类A/B本身不证明逐段原创。未执行网络、git、仿真、训练。', '',
        '## 分类定义', '', '| 类别 | 含义 |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in CATEGORIES.items()]
    lines += ['', '## 元数据统计', '', '| root_alias | 类别 | 记录数 | 普通文件字节 |', '|---|---:|---:|---:|']
    for alias in ROOTS:
        for cat in CATEGORIES:
            lines.append(f'| {alias} | {cat} | {by_root[alias][cat]} | {by_bytes[alias][cat]} |')
    lines += ['', '## 覆盖、hash和复制策略', '',
        '- 非缓存代码与大资产逐文件记录；symlink仅记自身且不跟随。排除子树以directory_skipped显式登记，其内部不遍历。字节统计不含这些子树，不能称整机磁盘总量。',
        '- SHA256只读取≤1MiB的小型源码/配置/许可；数据、权重、大文件只记元数据，不hash数十GB材料。敏感文件名不读取内容。',
        '- source_snapshots仅含小型研究scripts/configs/patches、必要shell与许可证；不复制assets、模型、数据、日志、内部交接文档、缓存、备份文件或编辑器配置。',
        '- OpenPI快照只包含相对本地openpi基线不同/新增的小源码与许可证，附恢复patch及逐文件baseline校验值；不是可独立安装的上游整仓。',
        '- secret扫描只输出被排除的root_alias/relative_path，不输出匹配内容；正则不能证明绝对无秘密，发布前仍需人工/最终复检。',
        '- 对匹配历史私人工作站路径前缀的小型候选文本，仅在暂存副本脱敏；snapshot_sha256记录脱敏后内容，sha256仍指源文件。',
        '- 原项目不写入、不删除。重复运行会刷新本工具生成的清单和已选快照；不会删除旧暂存，复用目录时应复检是否存在陈旧文件。', '',
        '## 本次快照', '', f'--snapshot：{args.snapshot}；复制文件数：{dict(copies)}。',
        f'清单记录数：{len(rows)}；secret-pattern排除：{len(secret_paths)}。', '']
    lines += ['- '+p for p in secret_paths]
    (report / 'PROJECT_INVENTORY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_csv(report / 'OPENPI_BASELINE_FILES.csv', baseline_rows,
              ['relative_path', 'baseline_exists', 'baseline_sha256', 'snapshot_sha256'])
    if args.snapshot:
        patch_dir = STAGING / 'source_snapshots' / 'openpi_hrc' / 'patches'
        patch_dir.mkdir(parents=True, exist_ok=True)
        (patch_dir / 'against_local_openpi_baseline.patch').write_text(''.join(patch_parts), encoding='utf-8')
    print(f'inventory_rows={len(rows)} copied={dict(copies)} secret_exclusions={len(secret_paths)}')
    for p in secret_paths:
        print('EXCLUDED_SECRET_CANDIDATE '+p)


if __name__ == '__main__':
    main()
