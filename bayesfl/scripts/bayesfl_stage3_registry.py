#!/usr/bin/env python3
"""BayesFL Stage 3: READ-ONLY registry of scripts, queues and configs.

Produces a text report and CSV inventories OUTSIDE the repository. It never
executes jobs or modifies, moves, deletes or rewrites project files.

Usage:
    cd ~/DungNDH/micl/bayesfl
    python scripts/bayesfl_stage3_registry.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
DEFAULT_ARCHIVE = PARENT / 'bayesfl_historical_archive'
DEFAULT_MANIFEST = PARENT / 'bayesfl_cleanup_DRY_RUN_20261026_194253.csv'
SCRIPT_ROOT = ROOT / 'scripts'

# These are safeguards/annotations, NOT an exhaustive list of all useful code.
CRITICAL_SCRIPTS = {
    'scripts/_run_gpu_queue.sh': 'CORE_QUEUE_EXECUTOR',
    'scripts/plot_ieee_multicast_results_mnist_s10_stage22.py': 'MAIN_PAPER_PLOTTER',
    'scripts/prepare_cifar_r150_sparse_resume.py': 'CIFAR_R150_RESUME_TOOL',
    'scripts/mnist_stage22_sparse_s10_seed0.py': 'MNIST_S10_EXPERIMENT_GENERATOR',
    'scripts/mnist_stage21_sparse_s5_seed3.py': 'MNIST_S5_EXPERIMENT_GENERATOR',
    'scripts/mnist_stage18_unclipped_fola.py': 'MNIST_DENSE_EXPERIMENT_GENERATOR',
    'scripts/bayesfl_cleanup_dry_run.py': 'CLEANUP_AUDIT_TOOL',
    'scripts/bayesfl_same_ssd_organizer.py': 'REVERSIBLE_ARCHIVE_MANAGER',
    'scripts/bayesfl_stage3_registry.py': 'STAGE3_AUDIT_TOOL',
}

TEXT_EXTENSIONS = {'.py', '.sh', '.yaml', '.yml', '.txt', '.json', '.md', '.toml', '.cfg', '.ini'}
PATH_PATTERN = re.compile(
    r'(?<![\w/])scripts/(?:configs|queues)/[A-Za-z0-9_./-]+\.(?:ya?ml|txt|json)',
    re.I,
)
RUN_NAME_PATTERN = re.compile(r'^\s*run_name\s*:\s*(.+?)\s*$', re.M)


def relative_string(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def under(path: Path, ancestor: Path) -> bool:
    return path == ancestor or ancestor in path.parents


def stamp(st_mtime: float) -> str:
    return datetime.fromtimestamp(st_mtime).strftime('%Y-%m-%d %H:%M:%S')


def read_text_small(path: Path, limit: int = 3_000_000) -> str:
    try:
        if path.stat().st_size > limit:
            return ''
        return path.read_text(encoding='utf-8', errors='replace')
    except (OSError, UnicodeError):
        return ''


def load_approved_manifest(manifest: Path):
    """Exact output path -> (category, action), if earlier plan is present."""
    info = {}
    if not manifest.is_file():
        return info, ['Prior dry-run CSV missing; archive categories cannot be cross-checked.']
    warnings = []
    with manifest.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rel = row.get('relative_path', '').strip()
            if not rel.startswith('outputs/') or rel.count('/') != 1:
                warnings.append('Unexpected path in previous plan: ' + rel)
                continue
            if rel in info:
                warnings.append('Duplicate previous-plan path: ' + rel)
            info[rel] = (row.get('category', ''), row.get('action', ''))
    return info, warnings


def resolve_output_reference(ref: str, manifest: dict, archive: Path):
    """Return origin status. Archive path is grouped by manifest category."""
    ref = ref.strip().strip('"\'')
    if not ref:
        return 'MISSING_REFERENCE', ''
    p = Path(ref)
    if not p.is_absolute():
        p = ROOT / p
    # Don't call resolve() here: it would conceal broken symlinks/relative paths.
    p = Path(os.path.abspath(p))
    active_outputs = ROOT / 'outputs'
    if under(p, active_outputs):
        try:
            relative = p.relative_to(ROOT)
        except ValueError:
            return 'OUTSIDE_PROJECT', str(p)
        if len(relative.parts) >= 2:
            run_rel = 'outputs/' + relative.parts[1]
            if p.exists():
                return 'ACTIVE_OUTPUT', str(p)
            if run_rel in manifest:
                category, action = manifest[run_rel]
                if action == 'ARCHIVE_CANDIDATE':
                    archived = archive / category / relative.parts[1]
                    if len(relative.parts) > 2:
                        archived = archived.joinpath(*relative.parts[2:])
                    if archived.exists():
                        return 'ARCHIVED_OUTPUT_REFERENCE', str(archived)
        return 'MISSING_OUTPUT_REFERENCE', str(p)
    if under(p, archive):
        return ('ARCHIVED_OUTPUT_REFERENCE' if p.exists() else 'MISSING_ARCHIVE_REFERENCE'), str(p)
    return ('OTHER_EXISTING_PATH' if p.exists() else 'OTHER_MISSING_PATH'), str(p)


def analyze_queue(line: str, manifest: dict, archive: Path):
    """Recognize exact record types accepted by _run_gpu_queue.sh."""
    if line.startswith('RESUME|'):
        parts = line.split('|')
        if len(parts) < 4:
            return 'RESUME', line, 'MALFORMED_RESUME', '', ''
        target = parts[1].strip()
        status, resolved = resolve_output_reference(target, manifest, archive)
        return 'RESUME', target, status, resolved, ''
    # Current executor treats every non-RESUME record as one config filepath.
    # Reject shell commands, which the executor would interpret as missing config.
    target = line.strip()
    if not (target.endswith('.yaml') or target.endswith('.yml')):
        return 'OTHER', target, 'NONSTANDARD_QUEUE_RECORD', '', ''
    p = Path(target)
    if not p.is_absolute():
        p = ROOT / p
    p = Path(os.path.abspath(p))
    status = 'CONFIG_FOUND' if p.is_file() else 'CONFIG_MISSING'
    canonical = relative_string(p)
    return 'CONFIG', target, status, str(p), canonical


def config_run_name(text: str) -> str:
    m = RUN_NAME_PATTERN.search(text)
    if not m:
        return ''
    value = m.group(1).split(' #', 1)[0].strip().strip('"\'')
    if re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        return value
    return ''  # Dynamic config; do not guess.


def source_category(rel: str) -> str:
    if rel in CRITICAL_SCRIPTS:
        return CRITICAL_SCRIPTS[rel]
    base = Path(rel).name.lower()
    if base.endswith(('.before_cifar_f1_patch', '.before_resume_support.sh')) or '.before_' in base or base.endswith('_old.py'):
        return 'HISTORICAL_VERSION_REVIEW'
    if base.startswith('plot_') or base.startswith('generate_plots'):
        return 'OTHER_PLOTTING_SCRIPT_REVIEW'
    if base.startswith('mnist_stage') or base.startswith('recover_') or base.startswith('prepare_'):
        return 'EXPERIMENT_OR_RECOVERY_SCRIPT_REVIEW'
    if base.startswith('run_') or base.startswith('smoke_'):
        return 'TRAINING_LAUNCHER_REVIEW'
    if base.startswith('bayesfl_') or base.startswith('migrate_'):
        return 'MAINTENANCE_TOOL_REVIEW'
    return 'GENERAL_SCRIPT_REVIEW'


def write_csv(path: Path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=PARENT,
                        help='Where to write reports (must be outside bayesfl)')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()

    if not SCRIPT_ROOT.is_dir() or not (ROOT / 'src').is_dir():
        parser.error('Unexpected project layout; place this file in bayesfl/scripts/.')
    report_dir = args.report_dir.expanduser().resolve()
    if under(report_dir, ROOT):
        parser.error('--report-dir must be OUTSIDE bayesfl; no project modifications allowed.')
    report_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive.expanduser().resolve()
    manifest, warnings = load_approved_manifest(args.manifest.expanduser())

    files = sorted(
        (p for p in SCRIPT_ROOT.rglob('*') if p.is_file() and not p.is_symlink()
         and '__pycache__' not in p.parts),
        key=lambda p: p.as_posix()
    )
    all_scripts = []
    configs = []
    queues = []
    others = []
    text_by_path = {}
    for p in files:
        rel = relative_string(p)
        if p.suffix.lower() in TEXT_EXTENSIONS:
            text_by_path[rel] = read_text_small(p)
        if p.parent == SCRIPT_ROOT:
            all_scripts.append(p)
        elif '/configs/' in rel:
            configs.append(p)
        elif '/queues/' in rel and p.suffix.lower() == '.txt':
            queues.append(p)
        else:
            others.append(p)

    # Reference detection is intentionally conservative: literal references only.
    mentions = defaultdict(set)
    for rel, content in text_by_path.items():
        for match in PATH_PATTERN.finditer(content):
            target = match.group(0)
            if target != rel:
                mentions[target].add(rel)
    for p in all_scripts:
        target = relative_string(p)
        basename = p.name
        for other_rel, text in text_by_path.items():
            if other_rel != target and basename in text:
                mentions[target].add(other_rel)

    active_runs = sorted((p.name for p in (ROOT / 'outputs').iterdir() if p.is_dir()),
                         key=str) if (ROOT / 'outputs').is_dir() else []
    archived_runs = [(path[len('outputs/'):], cat) for path, (cat, action) in manifest.items()
                     if action == 'ARCHIVE_CANDIDATE']
    config_rows = []
    for p in configs:
        rel = relative_string(p)
        st = p.stat()
        text = text_by_path.get(rel, '')
        run_name = config_run_name(text) if p.suffix.lower() in ('.yaml', '.yml') else ''
        live = []
        historical = []
        if run_name:
            live = [x for x in active_runs if x == run_name or x.startswith(run_name + '_')]
            historical = [name for name, _ in archived_runs
                          if name == run_name or name.startswith(run_name + '_')]
        output_status = ('ACTIVE_AND_ARCHIVED' if live and historical else
                         'ACTIVE_MATCH' if live else
                         'ARCHIVED_MATCH' if historical else
                         'NO_STATIC_RUN_MATCH' if run_name else
                         'NO_LITERAL_RUN_NAME')
        config_rows.append({
            'path': rel, 'type': p.suffix.lstrip('.').lower() or 'other',
            'bytes': st.st_size, 'modified': stamp(st.st_mtime),
            'run_name': run_name, 'output_status': output_status,
            'active_matches': len(live), 'archived_matches': len(historical),
            'literal_referencers': len(mentions.get(rel, ())),
            'queue_references': 0, 'review': 'REVIEW_NOT_DELETE',
        })

    job_rows = []
    queue_rows = []
    config_queue_refs = Counter()
    for p in queues:
        rel = relative_string(p)
        raw_lines = text_by_path.get(rel, '').splitlines()
        records = []
        for lineno, raw in enumerate(raw_lines, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            kind, target, status, actual, canonical = analyze_queue(line, manifest, archive)
            records.append(status)
            if kind == 'CONFIG' and status == 'CONFIG_FOUND':
                config_queue_refs[canonical] += 1
            job_rows.append({
                'queue': rel, 'line': lineno, 'kind': kind,
                'literal_target': target, 'status': status,
                'resolved_or_archive_location': actual,
            })
        st = p.stat()
        queue_rows.append({
            'path': rel, 'bytes': st.st_size, 'modified': stamp(st.st_mtime),
            'jobs': len(records), 'config_jobs': records.count('CONFIG_FOUND') + records.count('CONFIG_MISSING'),
            'resume_jobs': sum(1 for j in job_rows if j['queue'] == rel and j['kind'] == 'RESUME'),
            'archived_output_refs': records.count('ARCHIVED_OUTPUT_REFERENCE'),
            'missing_config_refs': records.count('CONFIG_MISSING'),
            'missing_output_refs': records.count('MISSING_OUTPUT_REFERENCE'),
            'nonstandard_records': records.count('NONSTANDARD_QUEUE_RECORD') + records.count('MALFORMED_RESUME'),
            'literal_referencers': len(mentions.get(rel, ())),
            'review': 'EMPTY_QUEUE_REVIEW' if not records else
                      'BROKEN_OR_HISTORICAL_REVIEW' if any(
                          s in {'ARCHIVED_OUTPUT_REFERENCE', 'CONFIG_MISSING', 'MISSING_OUTPUT_REFERENCE',
                                'NONSTANDARD_QUEUE_RECORD', 'MALFORMED_RESUME'} for s in records)
                      else 'HISTORICAL_OR_REUSABLE_REVIEW',
        })

    for row in config_rows:
        row['queue_references'] = config_queue_refs[row['path']]

    script_rows = []
    for p in all_scripts:
        st = p.stat()
        rel = relative_string(p)
        literal_script_refs = [ref for ref in mentions.get(rel, ())]
        embedded = text_by_path.get(rel, '')
        embedded_paths = sorted(set(m.group(0) for m in PATH_PATTERN.finditer(embedded)))
        script_rows.append({
            'path': rel, 'classification': source_category(rel),
            'bytes': st.st_size, 'modified': stamp(st.st_mtime),
            'literal_referencers': len(literal_script_refs),
            'static_config_or_queue_mentions': len(embedded_paths),
            'review': 'PRESERVE_IN_PLACE' if rel in CRITICAL_SCRIPTS else 'REVIEW_NOT_DELETE',
        })

    # Inspect previous cleanup/organization status WITHOUT touching anything.
    placement = Counter()
    for rel, (category, action) in manifest.items():
        in_root = (ROOT / rel).is_dir()
        in_archive = (archive / category / Path(rel).name).is_dir() if action == 'ARCHIVE_CANDIDATE' else False
        if action == 'KEEP_PROTECTED':
            placement['protected_ok' if in_root else 'protected_missing'] += 1
        elif action == 'ARCHIVE_CANDIDATE':
            if in_root and in_archive: placement['archive_duplicate'] += 1
            elif in_archive: placement['archived_ok'] += 1
            elif in_root: placement['not_archived'] += 1
            else: placement['archived_missing'] += 1

    queue_issue_rows = [q for q in queue_rows if q['review'] == 'BROKEN_OR_HISTORICAL_REVIEW']
    no_queue_configs = sum(c['queue_references'] == 0 for c in config_rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f'bayesfl_STAGE3_SCRIPT_REGISTRY_{ts}'
    paths = {
        'report': report_dir / f'{prefix}.txt',
        'scripts': report_dir / f'{prefix}_scripts.csv',
        'configs': report_dir / f'{prefix}_configs.csv',
        'queues': report_dir / f'{prefix}_queues.csv',
        'jobs': report_dir / f'{prefix}_queue_jobs.csv',
    }
    write_csv(paths['scripts'], script_rows,
              ['path','classification','bytes','modified','literal_referencers','static_config_or_queue_mentions','review'])
    write_csv(paths['configs'], config_rows,
              ['path','type','bytes','modified','run_name','output_status','active_matches',
               'archived_matches','literal_referencers','queue_references','review'])
    write_csv(paths['queues'], queue_rows,
              ['path','bytes','modified','jobs','config_jobs','resume_jobs','archived_output_refs',
               'missing_config_refs','missing_output_refs','nonstandard_records','literal_referencers','review'])
    write_csv(paths['jobs'], job_rows,
              ['queue','line','kind','literal_target','status','resolved_or_archive_location'])

    lines = []
    def add(line=''):
        lines.append(str(line))
    add('BAYESFL STAGE-3 SCRIPT / CONFIG / QUEUE REGISTRY — READ-ONLY')
    add('='*105)
    add('Project: ' + str(ROOT))
    add('Archive: ' + str(archive))
    add('Historical run manifest: ' + str(args.manifest))
    add('No project files were moved, deleted, or modified by this script.')
    add('Literal text references are INCOMPLETE dependency evidence, NOT proof of unused files.')
    add('')
    add('SUMMARY')
    add('-'*105)
    add(f'Top-level script/tool files: {len(script_rows)}')
    add(f'Files under scripts/configs/: {len(config_rows)}')
    add(f'Queue .txt files: {len(queue_rows)}')
    add(f'Queue job records: {len(job_rows)}')
    add(f'Empty queue files: {sum(r["jobs"] == 0 for r in queue_rows)}')
    add(f'Queues needing review: {len(queue_issue_rows)}')
    add(f'Configs without a static queue reference: {no_queue_configs} (NOT equivalent to unused)')
    add(f'Archived run folders present: {placement["archived_ok"]} / 121 expected')
    add(f'Protected original folders present: {placement["protected_ok"]} / 42 expected')
    problems = {k: n for k, n in placement.items() if k not in {'archived_ok', 'protected_ok'} and n}
    if problems:
        add('WARNING — unexpected run placements: ' + str(problems))
    if warnings:
        add('WARNING — manifest: ' + '; '.join(warnings))
    add('')
    add('A. MAIN WORKFLOW SCRIPTS — KEEP IN ORIGINAL LOCATIONS')
    add('-'*105)
    for row in script_rows:
        if row['review'] == 'PRESERVE_IN_PLACE':
            add(f'{row["classification"]:35s} {row["path"]}')
    add('')
    add('B. OTHER TOP-LEVEL SCRIPTS — CLASSIFY MANUALLY; DO NOT DELETE')
    add('-'*105)
    for row in script_rows:
        if row['review'] != 'PRESERVE_IN_PLACE':
            add(f'{row["classification"]:35s} refs={row["literal_referencers"]:3d} '
                f'{row["path"]}')
    add('')
    add('C. QUEUE FILES — JOB COUNTS AND FLAGS')
    add('-'*105)
    for row in queue_rows:
        add(f'jobs={row["jobs"]:3d} archived={row["archived_output_refs"]:2d} '
            f'config_missing={row["missing_config_refs"]:2d} '
            f'output_missing={row["missing_output_refs"]:2d} '
            f'other={row["nonstandard_records"]:2d} '
            f'{row["review"]:30s} {row["path"]}')
    add('')
    add('D. QUEUE JOBS POINTING TO ARCHIVED OR MISSING TARGETS')
    add('-'*105)
    issue_statuses = {'ARCHIVED_OUTPUT_REFERENCE', 'CONFIG_MISSING', 'MISSING_OUTPUT_REFERENCE',
                      'NONSTANDARD_QUEUE_RECORD', 'MALFORMED_RESUME', 'MISSING_ARCHIVE_REFERENCE'}
    for job in job_rows:
        if job['status'] in issue_statuses:
            add(f'{job["queue"]}:{job["line"]} [{job["status"]}] '
                f'{job["literal_target"]}')
    add('')
    add('E. CONFIG FILES — LITERAL run_name AND OUTPUT MATCHING')
    add('-'*105)
    for row in config_rows:
        add(f'queued={row["queue_references"]:2d} {row["output_status"]:20s} '
            f'run_name={row["run_name"] or "-"}  {row["path"]}')
    add('')
    add('F. SAFETY NOTES AND NEXT ACTION')
    add('-'*105)
    add('1. Keep src/, tests/, docs/, outputs/partitions/, current paper outputs and main plotter.')
    add('2. Do not move scripts/configs/ or scripts/queues/ yet; queues can contain relative paths.')
    add('3. Old script/queue names, no literal references, or missing run_name do NOT prove safe deletion.')
    add('4. Archived-output references may be restored with the Stage-2 same-SSD organizer if needed.')
    add('5. Send this TXT report for a more precise KEEP / HISTORICAL / OPTIONAL-CLEANUP review.')
    paths['report'].write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('=' * 75)
    print('STAGE-3 BAYESFL REGISTRY COMPLETE — NO PROJECT CHANGES')
    print('=' * 75)
    print(f'Scripts/config files/queues: {len(script_rows)} / {len(config_rows)} / {len(queue_rows)}')
    print(f'Queue jobs: {len(job_rows)}; flagged queues: {len(queue_issue_rows)}')
    print(f'Archived runs: {placement["archived_ok"]}/121; protected: {placement["protected_ok"]}/42')
    if problems or warnings:
        print('WARNINGS: report contains placement or manifest warnings; review them.')
    print('')
    for name, path in paths.items():
        print(f'{name:8s}: {path}')
    print('')
    print('UPLOAD THE TXT REPORT for classification and safe future organization.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
