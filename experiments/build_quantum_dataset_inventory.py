"""Build a provenance-first inventory of acquired quantum SE/QML sources."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/external_quantum_datasets'
OUT.mkdir(parents=True, exist_ok=True)


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ''


def artifact(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        return {'path': str(path.relative_to(ROOT)), 'bytes': None, 'sha256': None}
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def files_below(path: Path) -> tuple[int, int]:
    count = size = 0
    if path.exists():
        for base, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if d != '.git']
            for name in names:
                try:
                    size += (Path(base) / name).stat().st_size
                    count += 1
                except OSError:
                    pass
    return count, size


bugs4q = ROOT / 'data/external_quantum_datasets/bugs4q_full/official_framework'
bugs4q_db = ROOT / 'data/external_quantum_datasets/bugs4q_full/official_database'
bugs4q_na = ROOT / 'data/external_quantum_datasets/bugs4q_na'
qol = ROOT / 'data/external_quantum_datasets/qolumbina'
qol_emp = qol / 'extracted/qolumbina-empirical-study-main'
qol_infra = qol / 'qolumbina-infrastructure.zip'
qsim = ROOT / 'data/external_quantum_datasets/qsimbench/dataset'
qsim_files, qsim_bytes = files_below(qsim)
qmut_data = ROOT / 'data/qmutbench_full/data'
lintq = ROOT / 'data/external_quantum_datasets/lintq'
lintq_archive = lintq / 'raw/artifact_lintq_dataset.tar'
lintq_manifest = lintq / 'extracted/_extraction_manifest.json'
platform = ROOT / 'data/external_quantum_datasets/quantum_platform_bugs_223'
platform_zip = ROOT / 'data/external_quantum_datasets/quantum_platform_bugs_223_source.zip'
qml = ROOT / 'data/external_quantum_datasets/qml_platform_bugs'
qml_zip = ROOT / 'data/external_quantum_datasets/qml_platform_bugs_source.zip'
bug_pattern = ROOT / 'data/external_quantum_datasets/bug_pattern'
robust = ROOT / 'data/Bugs4Q_Robust_replication'

lintq_status = 'Extracted and verified' if lintq_manifest.exists() else (
    'Downloaded and MD5-verified; extraction pending' if lintq_archive.exists() else 'Pending')
records = [
    {
        'dataset': 'Bugs4Q framework (official Qiskit core)',
        'source': 'https://github.com/Z-928/Bugs4Q-Framework; https://zenodo.org/records/8148982',
        'status': 'Downloaded, extracted, verified', 'version': git_commit(bugs4q),
        'size': f'{files_below(bugs4q)[0]} files; git checkout',
        'records': '42 Qiskit bug issues; 42 buggy/fixed pairs; 84 source snapshots',
        'labels': 'Buggy program / repaired fixed program; issue group is the split unit',
        'formats': 'Python, CSV, diff text, reproduction tests',
        'task': 'Supervised buggy-versus-fixed source-state classification; grouped by bug issue',
        'license': 'Framework Apache-2.0; Zenodo database CC-BY-4.0; subject snippets may carry original upstream terms',
        'limitations': 'Small grouped corpus; 31-issue subset previously used overlaps 31/42; do not treat paired snapshots as independent',
        'next_action': 'Prepare all 42 issue groups for later training',
        'path': str(bugs4q.relative_to(ROOT)), 'hashes': json.dumps({'git_commit': git_commit(bugs4q)}),
    },
    {
        'dataset': 'Bugs4Q database (broader source catalog)',
        'source': 'https://github.com/Z-928/Bugs4Q; https://zenodo.org/records/8148982',
        'status': 'Downloaded, extracted, verified', 'version': git_commit(bugs4q_db),
        'size': f'{files_below(bugs4q_db)[0]} tracked files',
        'records': 'Contains Qiskit GitHub/Q&A examples plus Q# and Cirq; issue catalog overlaps framework and Bugs4Q-NA; canonical dedup count pending',
        'labels': 'Buggy/fixed examples; issue-source and bug-type metadata where present',
        'formats': 'Python, Q#, Jupyter, CSV, images',
        'task': 'Issue-grouped source-state and bug-pattern learning after deduplication',
        'license': 'Apache-2.0 repo; original program fragments may retain upstream terms; Zenodo database CC-BY-4.0',
        'limitations': 'The README mixes issue sources and duplicate Qiskit issue references; do not sum table rows as unique bugs',
        'next_action': 'Canonicalize issue identifiers against framework and NA repositories',
        'path': str(bugs4q_db.relative_to(ROOT)), 'hashes': json.dumps({'git_commit': git_commit(bugs4q_db)}),
    },
    {
        'dataset': 'Bugs4Q-NA', 'source': 'https://github.com/Z-928/Bugs4Q-NA',
        'status': 'Downloaded, extracted, verified', 'version': git_commit(bugs4q_na),
        'size': f'{files_below(bugs4q_na)[0]} tracked files',
        'records': '48 Python buggy/fixed pair directories in repository layout; overlaps legacy catalog and requires issue-ID deduplication',
        'labels': 'Buggy / fixed; issue identifiers and categories in README',
        'formats': 'Python, Q#, Jupyter, markdown',
        'task': 'Supervised source-state and issue-category tasks after deduplication',
        'license': 'Apache-2.0 repository; third-party source fragments retain upstream conditions',
        'limitations': 'Repository announces new Qiskit additions but its catalog includes older issue rows; unique increment beyond legacy is not yet established',
        'next_action': 'Diff issue identifiers against full Bugs4Q database before modeling',
        'path': str(bugs4q_na.relative_to(ROOT)), 'hashes': json.dumps({'git_commit': git_commit(bugs4q_na)}),
    },
    {
        'dataset': 'Bugs4Q-Robust reproducibility package',
        'source': 'https://github.com/kusumotolab/quantum-defect-reproducibility',
        'status': 'Downloaded, extracted, verified locally', 'version': git_commit(robust),
        'size': f'{files_below(robust)[0]} files',
        'records': '37 issue artifacts; 1,110 aggregate execution rows and raw repeated logs in the replication package',
        'labels': 'Buggy/fixed test outcomes; Complete Success / Partial Success / Failure in aggregate tables',
        'formats': 'Python, CSV, logs, Docker/configuration files',
        'task': 'Test outcome prediction or outcome analysis; issue-group split required',
        'license': 'Experiment code MIT; generated results CC-BY-4.0; derived subject programs excluded from package license',
        'limitations': 'Reproducibility outcomes vary by Qiskit version; many failures are dependency-related',
        'next_action': 'Normalize per-issue execution results and link to Bugs4Q source IDs',
        'path': str(robust.relative_to(ROOT)), 'hashes': '',
    },
    {
        'dataset': 'QMutBench', 'source': 'https://enautmendi.github.io/QMutBench/; https://arxiv.org/abs/2604.15870',
        'status': 'Existing local dataset and metadata verified', 'version': 'local checked-out source snapshot',
        'size': 'CSV metadata 60,636,991 bytes; dataset source files already present locally',
        'records': '723,079 mutant metadata rows across 300+ original circuits; 102,783 survival-group rows',
        'labels': 'Exact Killed-field values: WOO=368,568; Survivor=348,299; OPO=6,212. Source docs do not expand WOO/OPO codes.',
        'formats': 'QASM, CSV, HTML/JSON metadata',
        'task': 'Mutation outcome classification; keep the production binary task separate from any derived multiclass target',
        'license': 'Repository GPL-3.0; dataset-specific reuse terms not clearly separated in local README',
        'limitations': 'Do not expand undocumented label codes; mutant groups must stay within a single split',
        'next_action': 'Maintain current label semantics and group by original circuit',
        'path': str(qmut_data.relative_to(ROOT)), 'hashes': '',
    },
    {
        'dataset': 'Qolumbina empirical study artifact',
        'source': 'https://zenodo.org/records/20987445',
        'status': 'Downloaded, MD5-verified, extracted', 'version': 'Zenodo v1, 2026-06-28',
        'size': '29,870,691-byte archive; 15,880 extracted files; 201,639,269 bytes uncompressed',
        'records': '182 seeded buggy variants across 26 programs per artifact README; execution/test-report JSONs and specifications included',
        'labels': 'Seeded mutant variants and named mutation operators; original baseline implementations are in separate infrastructure artifact',
        'formats': 'Python, JSON, Markdown, PDF, LaTeX, CSV',
        'task': 'Mutation detection / test failure analysis; supervised code classification needs matching non-mutated baselines',
        'license': 'Apache-2.0 empirical-study repository; source projects retain attribution/terms',
        'limitations': 'Not a real-bug dataset; infrastructure archive is separately cited and not currently downloadable at cited DOI',
        'next_action': 'Use variant/report labels only after acquiring or reconstructing the matching baseline safely',
        'path': str(qol_emp.relative_to(ROOT)),
        'hashes': json.dumps(artifact(qol / 'qolumbina-empirical-study-main.zip')),
    },
    {
        'dataset': 'Qolumbina reusable infrastructure',
        'source': 'https://doi.org/10.5281/zenodo.20987326; documentation: https://qolumbinadoc.netlify.app/',
        'status': 'Blocked: cited Zenodo record currently returns Page Not Found', 'version': 'cited v0.1.0',
        'size': 'No infrastructure archive retrieved',
        'records': 'Paper/docs specify 40 scalable Qiskit programs, program specifications, and unit tests',
        'labels': 'No real-world bug labels; Qolumbina empirical package separately contains seeded mutants',
        'formats': 'Expected Python/Qiskit source and tests',
        'task': 'Controlled testing; may support functionality/input-conditioned tasks but not real-bug classification',
        'license': 'Not verified because the cited artifact is unavailable',
        'limitations': 'Empirical study explicitly separates implementation into infrastructure repository; current record URL is inaccessible',
        'next_action': 'Revisit if authors publish a working repository/archive URL',
        'path': str(qol.relative_to(ROOT)), 'hashes': '',
    },
    {
        'dataset': 'QSimBench execution-level dataset',
        'source': 'https://github.com/GBisi/qsimbench; https://github.com/GBisi/qsimbench-dataset',
        'status': 'Downloaded, Git LFS checkout complete', 'version': git_commit(qsim),
        'size': f'{qsim_files} files; {qsim_bytes} bytes by working-tree file sizes',
        'records': '4,500 JSONL trace files, 900 JSON metadata files; README states 20,000 50-shot outcome batches per configuration',
        'labels': 'Measured bitstring counts plus algorithm, qubit count, backend, circuit, and noise metadata; no bug/fix labels',
        'formats': 'JSONL, JSON, Python package metadata',
        'task': 'Execution-output/noise/backend prediction or regression; group by circuit/configuration to prevent leakage',
        'license': 'AGPL-3.0',
        'limitations': 'Large execution traces; record-line total not yet counted across all 4,500 files; outcomes are not independent bug labels',
        'next_action': 'Build streaming index and validate representative trace schemas before modeling',
        'path': str(qsim.relative_to(ROOT)), 'hashes': json.dumps({'git_commit': git_commit(qsim)}),
    },
    {
        'dataset': 'Quantum platform bugs (Paltenghi & Pradel)',
        'source': 'https://github.com/MattePalte/Bugs-Quantum-Computing-Platforms; https://arxiv.org/abs/2110.14560',
        'status': 'Downloaded; all 11,788 files extracted with Windows-invalid path mappings recorded',
        'version': git_commit(platform), 'size': '157,031,819-byte source archive; 153,762,125 bytes extracted',
        'records': '223 manually inspected real-world platform bug cases; annotation CSV also contains false positives and sampling data',
        'labels': 'Bug/false-positive and bug type, component, symptom, pattern, comments, localization; before/after minimal fixes',
        'formats': 'CSV, Python and other source files, notebooks, markdown, PDFs',
        'task': 'Supervised bug type/component/pattern classification; group by fix commit and project',
        'license': 'MIT',
        'limitations': 'Windows extraction percent-encodes original `>` path characters; exact original names remain in source archive and mapping manifest',
        'next_action': 'Normalize annotation CSV and join minimal before/after files by human_id/commit',
        'path': str(platform.relative_to(ROOT)), 'hashes': json.dumps(artifact(platform_zip)),
    },
    {
        'dataset': 'Quantum ML framework issue collection',
        'source': 'https://github.com/Z-928/QML-platform-bugs',
        'status': 'Downloaded and extracted; source ZIP CRC verified', 'version': git_commit(qml),
        'size': '1,480,120-byte source archive; 16 classified workbooks and 22 original-issue workbooks',
        'records': '15,984 rows across classified workbooks; 14,168 rows across original workbooks (per XLSX worksheet rows, before dedup)',
        'labels': 'Manual 0/1 agreement and quantum-related annotations, plus issue status/text and some issue-type labels',
        'formats': 'XLSX and nested ZIP archives',
        'task': 'Issue-level quantum-related and annotation-agreement classification; deduplicate across workbook views',
        'license': 'No license file located in the repository; issue text remains subject to platform/source terms',
        'limitations': 'Original and classified workbooks overlap; some exports have 999 non-header rows; de-duplication and schema reconciliation pending',
        'next_action': 'Normalize workbook schemas and deduplicate by issue URL',
        'path': str(qml.relative_to(ROOT)), 'hashes': json.dumps(artifact(qml_zip)),
    },
    {
        'dataset': 'LintQ quantum-program corpus',
        'source': 'https://github.com/sola-st/LintQ; https://doi.org/10.6084/m9.figshare.22350592',
        'status': lintq_status, 'version': git_commit(lintq),
        'size': '2,484,695,040-byte Figshare TAR; 89,640 files / 2,407,157,140 extracted bytes' if lintq_manifest.exists() else '2,484,695,040-byte Figshare TAR; expected 89,640 files / 2,407,157,140 extracted bytes',
        'records': '7,568 real-world quantum programs per LintQ paper; TAR also contains analysis artifacts and metadata',
        'labels': 'Static-analysis warning outputs and manually inspected true positives; dataset should not be treated as one flat binary label without mapping',
        'formats': 'Python source, CSV, CodeQL data, analysis outputs',
        'task': 'Program representation, warning detection, and rule-specific classification',
        'license': 'Figshare dataset CC-BY-4.0; LintQ software code has separate repository license',
        'limitations': 'Large corpus; extracted file count is not the same as program count; preserve per-program IDs',
        'next_action': 'Finish extraction, verify file count/size, identify label-bearing result files and program groups',
        'path': str(lintq.relative_to(ROOT)),
        'hashes': json.dumps(artifact(lintq_archive)) if lintq_archive.exists() else '',
    },
    {
        'dataset': 'Bug-Pattern examples',
        'source': 'https://github.com/Z-928/Bug-Pattern; https://github.com/Z-928/Bugs4Q',
        'status': 'Downloaded, extracted, verified', 'version': git_commit(bug_pattern),
        'size': f'{files_below(bug_pattern)[0]} tracked files; 9 example notebooks',
        'records': 'Eight top-level pattern codes documented (including incorrect gates, measurement, initialization, parameter and command issues); example count is small and overlaps Bugs4Q/Q&A',
        'labels': 'Pattern-code labels in README and example organization',
        'formats': 'Jupyter notebooks, README, Python support files',
        'task': 'Small pattern classification or rule-development set, grouped by source issue',
        'license': 'Apache-2.0 repository',
        'limitations': 'Examples are not independent of Bugs4Q/Q&A and are too small for standalone benchmark claims',
        'next_action': 'Deduplicate issue URLs against Bugs4Q database before reuse',
        'path': str(bug_pattern.relative_to(ROOT)), 'hashes': json.dumps({'git_commit': git_commit(bug_pattern)}),
    },
    {
        'dataset': 'QBugs',
        'source': 'https://arxiv.org/abs/2103.16968; https://zenodo.org/records/14288140; https://github.com/JanusQ/HornBro',
        'status': 'Downloaded, archive MD5 and CRC verified; QBugs circuit subset extracted',
        'version': 'HornBro Zenodo v3, 2024-12-06', 'size': '49,259,382-byte RAR; QBugs subset 15 files / 4,955,098 bytes',
        'records': '15 qcs.pkl scenario files; 100 circuits each; 750 correct/buggy pairs (1,500 circuit snapshots) as described by HornBro README',
        'labels': 'Original/correct=0, buggy=1; first and second halves are paired by index',
        'formats': 'Pickled Qiskit QuantumCircuit lists',
        'task': 'Grouped correct-versus-buggy circuit-state classification; split by scenario and pair index',
        'license': 'HornBro software MIT; original QBugs dataset-specific terms not separately established',
        'limitations': 'This is the QBugs derivative included by HornBro, not a separately verified issue-level catalog. Pickles require Qiskit 1.4.5 compatibility.',
        'next_action': 'Grouped QNN and matched classical evaluation completed; keep separate from Bugs4Q tasks',
        'path': 'data\\external_quantum_datasets\\qbugs_hornbro',
        'hashes': '{"archive_md5":"8836278a06f21411ccf35cb8ff872353"}',
    },
    {
        'dataset': 'Mute-Alg synthetic quantum mutations',
        'source': 'https://zenodo.org/records/14288140; https://github.com/JanusQ/HornBro',
        'status': 'Downloaded and extracted; circuit pickle deserialization blocked by Qiskit state incompatibility',
        'version': 'HornBro Zenodo v3', 'size': '169 qcs.pkl files; 324,105,511 bytes extracted',
        'records': 'Each qcs.pkl is documented as original and buggy circuit halves; exact usable pair count pending due pickle loader incompatibility',
        'labels': 'Documented original versus mutated buggy circuit states',
        'formats': 'Pickled Qiskit QuantumCircuit lists',
        'task': 'Synthetic correct-versus-buggy circuit-state classification if compatible deserialization is available',
        'license': 'HornBro software MIT; generated dataset-specific terms not separately established',
        'limitations': 'Restricted unpickling failed with Qiskit 1.4.5 `state is not a dictionary`; Qiskit 2.5 raised a circuit register compatibility error. No examples were counted or trained.',
        'next_action': 'Revisit with the Qiskit version used to generate the archive or a source-level export',
        'path': 'data\\external_quantum_datasets\\hornbro_mute_alg',
        'hashes': '{"source_archive_md5":"8836278a06f21411ccf35cb8ff872353"}',
    },
    {
        'dataset': 'Qiskit quantum issue-label corpus (12,910 issues / 4,984 annotated sample)',
        'source': 'https://arxiv.org/abs/2506.10397; reported Zenodo record search result https://zenodo.org/records/23647909',
        'status': 'Pending: reported Zenodo record/API currently returns cache miss or no metadata file',
        'version': '2025 paper; secondary record surfaced 2026-10-28', 'size': 'No archive retrieved',
        'records': 'Paper reports 12,910 issues across 36 Qiskit repositories and 4,984 manually annotated issues',
        'labels': 'Bug type, category, severity, quality attribute, quantum-specific subtype per paper',
        'formats': 'Reported JSON ZIP artifact was not verified/downloaded',
        'task': 'Issue-level multi-label/multitask classification if the actual artifact becomes accessible',
        'license': 'Not verified for the dataset artifact',
        'limitations': 'Search metadata reported a 574.2 MB Zenodo ZIP, but opening the record returned cache miss and the API request did not produce metadata',
        'next_action': 'Retry only after a stable public record/download URL is available', 'path': '', 'hashes': '',
    },
    {
        'dataset': 'Cross-ecosystem quantum issue corpus',
        'source': 'https://arxiv.org/abs/2608.03173',
        'status': 'Pending: paper reports data but no direct downloadable artifact located',
        'version': '2026 preprint', 'size': 'No archive retrieved',
        'records': 'Paper reports 12,910 Qiskit issues plus 4,613 issues across 11 non-Qiskit repositories',
        'labels': 'Bug type, category, severity, quality attributes and quantum-specific subtypes per paper',
        'formats': 'Not verified locally',
        'task': 'Cross-ecosystem issue classification if issue text and annotations are publicly released',
        'license': 'Not established',
        'limitations': 'No accessible artifact/download link verified in this campaign',
        'next_action': 'Revisit the paper artifact link when available', 'path': '', 'hashes': '',
    },
]

fields = ['dataset', 'source', 'status', 'version', 'size', 'records', 'labels', 'formats',
          'task', 'license', 'limitations', 'next_action', 'path', 'hashes']
csv_path = OUT / 'master_dataset_inventory.csv'
with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
json_path = OUT / 'master_dataset_inventory.json'
json_path.write_text(json.dumps({'generated_on': date.today().isoformat(), 'datasets': records}, indent=2), encoding='utf-8')
print(json.dumps({'csv': str(csv_path.relative_to(ROOT)), 'json': str(json_path.relative_to(ROOT)), 'datasets': len(records)}, indent=2))
