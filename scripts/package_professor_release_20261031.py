"""Create a report/evidence package without altering original research artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from scripts.verify_professor_package import check

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "LightSAKT2"
OLD = PROJECT / "reports/professor_results_20261030"
OUT = PROJECT / "reports/professor_release_20261031"
WORK = PROJECT / ".tools/professor_release_20261031"
BACKUP = SOURCE / "backups/professor_release_20261031"
NAME = "LightSAKT_question_difficulty_verified_results_20261031"
ARCHIVE = BACKUP / (NAME + ".zip")
DATASETS = ["ASSIST2009", "ASSIST2015", "Algebra2005"]
MODELS = ["SAKT", "LightSAKT", "LightSAKT+Difficulty"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reports():
    assert read(OUT / "verification_summary.json")["status"] == "PASS"
    assert read(OUT / "assist2015_official_release_audit.json")["identical_to_local_raw"]
    rows = read(OUT / "summary_results.json")
    individuals = read(OUT / "individual_seed_results.json")
    deltas = read(OUT / "absolute_auc_improvements.json")

    def value(row, metric):
        if not row["verified_seeds"]:
            return "UNAVAILABLE"
        return f"{row[metric + '_mean']:.6f} ± {row[metric + '_sd']:.6f}"

    table = ["| Dataset | Model | Validation AUC | Test AUC | Test accuracy | Test F1 | Seeds |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        metrics = [value(row, metric) for metric in ["validation_auc", "test_auc", "test_accuracy", "test_f1"]]
        table.append(f"| {row['dataset']} | {row['model']} | " + " | ".join(metrics) + f" | {row['verified_seeds']}/5 |")
    comparison = "\n".join(table)
    (OUT / "comparison_table.md").write_text(comparison + "\n")
    delta_table = ["| Dataset | Comparator | Validation AUC gain | Test AUC gain | Positive validation / test seeds |",
                   "|---|---|---|---|---|"]
    for row in deltas:
        if row["paired_seeds"]:
            a = f"{row['validation_auc_delta_mean']:+.6f} ± {row['validation_auc_delta_sd']:.6f}"
            b = f"{row['test_auc_delta_mean']:+.6f} ± {row['test_auc_delta_sd']:.6f}"
            positive = f"{row['validation_positive_seed_count']}/5 / {row['test_positive_seed_count']}/5"
        else:
            a = b = "UNAVAILABLE"; positive = "—"
        delta_table.append(f"| {row['dataset']} | {row['comparator']} | {a} | {b} | {positive} |")
    seed_table = ["| Dataset | Model | Seed | Validation AUC | Test AUC | Test accuracy | Test F1 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
    for row in individuals:
        values = [f"{row[key]:.9f}" if row["status"] == "VERIFIED" else "UNAVAILABLE"
                  for key in ["validation_auc", "test_auc", "test_accuracy", "test_f1"]]
        seed_table.append(f"| {row['dataset']} | {row['model']} | {row['seed']} | " + " | ".join(values) + " |")
    definitions = '''# Meaning of comparison columns

| Column | Definition |
|---|---|
| Validation AUC | Native recorded validation AUC at that seed's validation-selected checkpoint; the prior CPU replay reproduces it within the declared numerical tolerance. |
| Test AUC | AUC on the fixed held-out student test split, from CPU replay of the same selected checkpoint. It is not another validation score. |
| Test accuracy / F1 / precision / recall | Globally pooled valid next-response targets; fixed probability threshold 0.5. No threshold optimization. |
| BCE | Mean binary cross-entropy on valid targets, clipping probabilities to [1e-7,1-1e-7] for reporting. |
| Mean ± SD | Unweighted mean across five prescribed training seeds and SAMPLE standard deviation (ddof=1). No SD is estimated for unavailable results. |
| Validation AUC gain | For each matched seed: modified LightSAKT validation AUC MINUS the named comparator's validation AUC; then mean/sample SD of those paired differences. |
| Test AUC gain | The corresponding paired subtraction using held-out test AUC. Positive favors the modified model; units are absolute AUC, not relative percent. |
| Comparator | Either original causal LightSAKT or original causal SAKT, identified separately in every difference row. |
| Positive seed count | Number of paired differences strictly greater than zero; not a significance test or confidence interval. |
| UNAVAILABLE_CURRENT_PROTOCOL | No valid modified ASSIST2015 measurement, because independent question IDs cannot be obtained from the inspected release. Never a zero score or estimated baseline-equivalent score. |
| Checkpoint reference | External logical weight locator and SHA256. Portable configurations are included; weights and raw datasets are not in this reporting archive. |

Validation scores are recorded native scores; test columns consistently use the
2026-10-30 CPU replay (batch size 8). Both refer to exactly the same frozen
checkpoint. Current verification recalculates metrics from all 80 saved replay
arrays and checks all 40 checkpoint hashes/configurations; it performs no new
training or model inference. The original replay evidence is included unchanged.

Baseline final configurations have evaluate_test=true for test scoring AFTER
training/checkpoint selection. Difficulty training has evaluate_test=false and
separate post-selection CPU scoring. This flag does not alter training loss,
optimizer, early stopping or checkpoint selection. Selection is maximum validation
AUC for all models. The fixed 0.5 threshold applies to both splits.
'''
    (OUT / "METRIC_DEFINITIONS.md").write_text(definitions)
    blocker = '''# ASSIST2015 question-identifier recovery: unavailable under current protocol

The publisher-linked original `2015_100_skill_builders_main_problems.csv` was
retrieved and compared with the working raw file and preserved raw-data archive.
All three match byte-for-byte: 18,334,729 bytes, SHA256
`75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0`.
There are 708,631 rows and only four fields:
`user_id, log_id, sequence_id, correct`. The existing processed copy also has no
independent question identity. Project-local CSV/TSV headers and preserved raw
and processed archives were checked; no authentic event-to-question crosswalk
was found. File hashes and search boundaries are recorded in the JSON audits.

The official release covers 100 skill builders. The publisher's field dictionary
distinguishes question identity (`problem_id`) from a sequence/problem-set ID and
an interaction-log record ID. Thus `sequence_id` is a skill-builder/problem-set
proxy used by the established model input, not an individual question. The older
report's word “skill” describes that operational proxy; preprocessing is unchanged.
The raw file has 708,631 distinct log IDs and 100 sequence IDs. A unique event ID
does not identify which repeated question was answered across events/students.
The current maintained pyKT preprocessor likewise supplies no question sequence.
[Publisher release](https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data),
[publisher field dictionary](https://sites.google.com/site/assistmentsdata/an-explanation-on-how-to-interpret-our-data-sets),
[pyKT primary source](https://api.github.com/repos/pykt-team/pykt-toolkit/contents/pykt/preprocess/assist2015_preprocess.py).

No skill/sequence IDs, log IDs, row numbers, hashes, responses or ordering
heuristics were converted into invented question IDs. Different underlying
question assignments could generate the same released four-column records;
unique reconstruction is not supported by these fields alone. Another-year
dataset or an unrelated release would not preserve the current benchmark.

The bounded conclusion is that authentic recovery was unsuccessful from the
inspected accessible sources. This is not a claim that the provider's internal
database lacks the mapping. A provider-supplied `log_id -> problem_id` export for
the exact original records would be needed. Verify event uniqueness, student
alignment, coverage of every retained interaction and absence of response-based
inference, then freeze the feature protocol before any new execution.

All five ASSIST2015 difficulty slots are UNAVAILABLE_CURRENT_PROTOCOL, not queued
training jobs. The SAKT and LightSAKT results remain valid and reusable. Current
preprocessing retains 683,801 binary interactions after excluding 24,830
nonbinary-response rows; no cleaning, row order, vocabulary or split was changed.
'''
    (OUT / "ASSIST2015_LIMITATION.md").write_text(blocker)
    settings = read(OLD / "training_configurations.json")
    config_table = ["| Dataset | Model | Dimension | Heads | Dropout | Learning rate | Neural parameters |",
                    "|---|---|---:|---:|---:|---:|---:|"]
    for row in settings:
        cfg = row["model_config"]
        config_table.append(f"| {row['dataset']} | {row['model']} | {cfg.get('embed_dim',cfg.get('d'))} | {cfg.get('num_heads',cfg.get('num_attn_heads'))} | {cfg['dropout']} | {row['learning_rate']} | {row['parameters']} |")
    report = f'''# LightSAKT + Question Difficulty: verified professor report

31 October 2026. **40 measured model-seed results; five ASSIST2015 difficulty
slots unavailable under the current protocol. No new GPU experiment executed.**

## Findings and suitability to share

The existing ASSIST2009 seed-42 validation change from 0.750803537 to 0.781189972
is supported by the preserved checkpoints and replay evidence. It is a validation
result, not a test score. Across five seeds on each completed dataset, the
modified model improves both validation and held-out test AUC in every matched
seed versus both SAKT and LightSAKT. Mean test accuracy and F1 also improve.

**These results are sufficient to share as a verified research progress report.**
They do not establish publication readiness, complete three-dataset improvement,
attention-kernel superiority, or performance in chronological deployment. The
missing dataset experiment and feature-matched control remain material gaps.

## Complete comparison status

{comparison}

All displayed uncertainties are sample SD across training seeds
42, 123, 2026, 3407 and 7777. Accuracy/F1 here are TEST metrics. Validation and
test AUC are separate columns. Exact per-seed validation/test accuracy, F1,
precision, recall and BCE are in individual_seed_results.csv/JSON; summary files
contain mean/SD for every metric. No missing measurement is manufactured.

## Paired improvements: modified LightSAKT minus named comparator

{chr(10).join(delta_table)}

The SD above describes paired per-seed differences, not subtraction of group SDs.
Positive means higher AUC; these are absolute AUC units. Full definitions and the
native-validation versus CPU-test provenance are in METRIC_DEFINITIONS.md.

## Architecture and information supplied

The baseline LightSAKT architecture has one causal normalized ELU+1 linear
attention block, learned skill/response and position embeddings, residual layer
normalization, a d -> d/2 -> d ReLU feed-forward network and sigmoid readout.
SAKT retains its native causal softmax attention and d -> d -> d feed-forward
network. “Original” denotes the established corrected causal implementations
with original trainable architecture/parameter shapes. The preserved historical
original_snapshot had attention/masking/test-selection defects; its unverified
historical scores are excluded. This distinction must accompany dissemination.

Modified LightSAKT changes only its readout:

`p = sigmoid(original_logit + alpha * feature)`, with zero-initialized learned alpha.

The smoothed skill success rate is `(successes+1)/(count+2)`; the question rate is
`(question_successes + 10*skill_rate)/(question_count+10)`; the feature is their
logit difference. Positive values represent relatively easier questions. The
encoder, depth, attention mechanism and per-dataset LightSAKT settings are unchanged.
The one extra neural parameter is accompanied by fitted item statistics: the
training corpus contains 16,511 question aggregates on ASSIST2009 and 119,581
target-step aggregates on Algebra2005. These are extra information/estimation
capacity and are not counted as neural weights.

## Leakage, causal history and unavailable data

For a training row, the complete student's responses are subtracted from both
skill and question statistics, including its current and future responses.
Validation/test feature fitting uses training students' labels only. Query
metadata is transformed without current outcome labels. Unseen questions receive
zero correction through the smoothed skill-rate fallback. Fresh real-data
held-out-label and whole-own-student-label perturbations passed on both datasets,
including a responsive other-student positive control. All original feature
sidecars were reproduced; no feature, split or label was edited.

The encoder remains causal. Earlier observed held-out responses form history
for later predictions, as in the baseline, but future held-out responses and the
current target are excluded. All expected per-student n-1 targets are present
in order without padding, duplication or discarded difficult examples.

Statistics fit the COMPLETE training corpus. The earlier audit found later-order
training observations for 95.045% of ASSIST2009 validation attempts. Offline
student-disjoint fitting is the stated protocol; forward-time availability has
not been demonstrated. Algebra predictions condition on the known target
problem/step identity, consistent with the documented benchmark fields.

For ASSIST2015, the publisher's original file is byte-identical to the local
four-column release and raw backup. No independent question ID or authentic
log-to-question crosswalk was found. Sequence IDs are problem-set/skill-builder
proxies and log IDs identify events; neither is an item-ID substitute. All five
modified slots are unavailable for this protocol. See ASSIST2015_LIMITATION.md
for evidence, primary links, exact search bounds and requirements to unblock.

## Dataset and evaluation consistency

All comparisons use the existing fixed student-disjoint 70/10/20 partitions,
split seed42, training-only vocabulary and identical per-dataset preprocessing.
ASSIST2009/ASSIST2015 sequences have length100; Algebra length200. Chunking
preserves all n-1 targets with a one-interaction boundary overlap. The full
KC(Default) composite is one Algebra skill; its raw training subset is locally
student-split. Its test column is not the official challenge test. Dataset
counts and fingerprints are in unchanged evidence_20261030/dataset_metadata.json.

{chr(10).join(config_table)}

All runs use batch64, float32, no AMP/TF32, Adam betas(0.9,0.999), epsilon1e-8,
weight decay0, maximum50epochs, patience5 and strict validation-AUC checkpoint
selection. SAKT/LightSAKT retained their separately selected settings from the
same36-point validation grid. Difficulty models inherit their native LightSAKT
settings, without retuning. Each modified configuration was compared with its
matched baseline configuration and all40 saved checkpoint configurations.

Baseline final configs enable test evaluation AFTER training/checkpoint selection;
modified training disables it and uses a separate CPU evaluation. The table
consistently uses the previously saved CPU test replay, batch8, globally pooled
masked targets and fixed0.5 threshold. Validation values are the native selected
epoch scores, reproduced within AUC/BCE2e-6 and accuracy/F1 tolerance1e-4 by CPU
replay. The flag difference changes evaluation scheduling, not optimization or
checkpoint selection. No test-based tuning or new inference was performed here.

## Verification and provenance

Fresh checks read all77 files in the previous report directory, verify all40
checkpoint hashes, finite weights, configurations, shapes, parameter counts and
validation-best epochs, and recalculate metrics from all80 saved validation/test
prediction arrays. Independent rank-sum AUC and manual confusion counts agree.
Ten difficulty optimizer/RNG/shuffle-state audits and338 protected-file hashes
pass. Every aggregate mean/sample SD and all paired differences match per-seed
evidence. Original reports, checkpoints, raw files and scientific sources remain
unchanged; verification_summary.json and original_report_fingerprints.json record
the boundaries. This is a fresh audit of the prior checkpoint replay, not a new
independent training campaign or an additional inference measurement.

Results are anchored at c286bcbc; the preceding handoff is49982fa7. Three local
incremental Git bundles verify, with their prerequisite baseline commits
preserved. The bundle chain is not an independent full-history repository.
No remote push was attempted or claimed; the recorded CLI authentication blocker
remains. HiTSKT and unrelated workloads were not modified.

## Individual seed results

{chr(10).join(seed_table)}

## Limitations and next required comparison

Five initializations on one split per dataset do not establish population-level
uncertainty or cross-cohort replication. Earlier validation reuse introduces
selection exposure. These are local dataset variants and globally pooled
interaction metrics, so direct comparison with unrelated published protocols
would be inappropriate. This analysis is descriptive, with no post-hoc
significance or state-of-the-art claim. The released test split must not guide
further tuning.

The key missing control is **SAKT + the identical question-success correction**:
same item information, smoothing, whole-student exclusion, training-only held-out
fitting, scalar readout, partitions, five seeds, loss/masking/budget and validation
selection, while preserving SAKT's own native encoder and previously selected
settings. Compare its gain over SAKT with the difficulty gain over LightSAKT,
and compare both feature-matched models. That isolates input information from
encoder-family differences more clearly than the current richer-input comparison.
It is already prepared but unexecuted and requires a separately defined resource
budget. Do not repeat the completed LightSAKT seeds in the older control plan.

## Portable reporting package

The portable archive is `{ARCHIVE}`. It includes this report and summary, all
CSV/JSON tables, copied original report evidence, scientific source snapshots,
40 frozen configs, provenance, checkpoint SHA256 references and a standalone
standard-library verifier. Manifest paths are relative and archive readback,
extraction, integrity and aggregation checks are performed. External local paths
are provenance locators, not portability dependencies for reading/verifying tables.

Raw datasets, per-student prediction/target arrays, model weights, credentials and
transient download URLs are excluded. Checkpoint references and data fingerprints
allow exact matching to the preserved local artifacts, but the reporting archive
is not a self-contained training or model-inference bundle. See README.md and
the external archive_verification.json for its exact integrity scope/checksum.
'''
    (OUT / "PROFESSOR_REPORT.md").write_text(report)
    summary = '''# Professor-ready research summary — 31 October 2026

We verified the LightSAKT question-difficulty results across five random seeds
on ASSIST2009 and Algebra2005. The seed-42 ASSIST2009 validation AUC change
0.7508 -> 0.7812 reproduces from saved artifacts. Mean held-out test AUC improves
from0.751056 to0.779359 on ASSIST2009 (+0.028303) and from0.810590 to0.836871 on
Algebra2005 (+0.026281). All five matched seeds improve validation and test AUC
on both datasets, versus both original LightSAKT and SAKT. Mean test F1 and
accuracy also improve. Tables report sample standard deviations and all seeds.

ASSIST2015's official original CSV exactly matches the preserved four-column
release. It contains interaction IDs and skill-builder/problem-set IDs, but no
independent question ID or recovered item crosswalk. Its five difficulty-model
results are unavailable under the current protocol; baseline results remain valid.

Leakage and checkpoint/configuration/split/masking/metric checks passed for the
stated offline student-disjoint protocol. No GPU experiments were repeated.
The modified models receive richer item information; a feature-matched SAKT
control remains necessary to isolate that contribution. Complete training-corpus
statistics do not demonstrate forward-time deployment performance.

These findings are appropriate to share as verified research progress, with the
ASSIST2015 gap and limitations stated. They do not yet justify publication
readiness, three-dataset superiority, or an attention-architecture advantage.
'''
    (OUT / "PROFESSOR_SUMMARY.md").write_text(summary)
    (OUT / "README.md").write_text('''# Portable LightSAKT results package

Start with [PROFESSOR_REPORT.md](PROFESSOR_REPORT.md), the shorter
[PROFESSOR_SUMMARY.md](PROFESSOR_SUMMARY.md), and [comparison_table.md](comparison_table.md).
Metric meanings: [METRIC_DEFINITIONS.md](METRIC_DEFINITIONS.md).
Missing dataset evidence: [ASSIST2015_LIMITATION.md](ASSIST2015_LIMITATION.md).

Forty measured rows and five unavailable ASSIST2015 difficulty slots are listed
in individual_seed_results.csv/JSON. summary_results.csv/JSON includes mean and
sample SD of validation/test AUC, accuracy, F1, precision, recall and BCE.
absolute_auc_improvements.csv and paired_seed_deltas.csv distinguish validation
from test effects and label each comparator. No missing score is estimated.

After extraction, run `python verify_package.py` from this directory (Python3.9+,
standard library only). It validates SHA256 and file coverage, recomputes all
aggregate metrics and paired AUC differences, and checks CSV/JSON/config/reference
consistency. It performs no network access, training, checkpoint loading or inference.
MANIFEST.json excludes itself from its payload hashes; the external ZIP checksum
anchors the entire archive. This is integrity/consistency verification, not a
digital signature or an independent rerun of model evaluation.

evidence_20261030/ is a byte-preserved copy of the prior reports. Its historical
PENDING label is superseded by today's UNAVAILABLE_CURRENT_PROTOCOL conclusion.
The new report does not alter those old files or claim new measurements.
checkpoint_references.csv/JSON contains40 external weight references, selected
epochs, SHA256s, dataset/split hashes and links to included configurations/.
source_local_path and original configuration absolute paths document the source
machine. They are not expected to exist on a recipient's computer.

Weights, raw/processed datasets and per-student prediction/target arrays are not
included. To rerun inference, independently obtain the matching authorized data,
resolve each logical checkpoint reference to its exact SHA256-matching weight
file, reconstruct the train-only difficulty transform using the frozen algorithm,
and use the included source/config snapshots or original result repository.
This packet alone is sufficient to read/recheck the reported aggregations, not
to retrain or reproduce checkpoint inference without those external artifacts.

Local Git bundle verification is documented in git_bundle_verification.json.
The bundles are incremental and depend on preserved base commits; they are not
included in this report package. No remote push or offsite backup is claimed.
''')


def main():
    reports()
    expected_old = read(OUT / "original_report_fingerprints.json")
    assert expected_old == {str(p.relative_to(OLD)): sha(p) for p in OLD.rglob("*") if p.is_file()}
    BACKUP.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        raise FileExistsError("Preserve existing delivery ZIP; do not overwrite it")
    payload = WORK / "package" / NAME
    if payload.exists():
        raise FileExistsError("Preserve existing package tree; use a new release identifier")
    payload.mkdir(parents=True)
    for path in OUT.iterdir():
        if path.name == "archive_verification.json":
            continue
        if path.is_file():
            shutil.copy2(path, payload / path.name)
    shutil.copytree(OUT / "configurations", payload / "configurations")
    shutil.copytree(OLD, payload / "evidence_20261030")
    shutil.copy2(PROJECT / "scripts/verify_professor_package.py", payload / "verify_package.py")
    code_root = payload / "scientific_source"
    for path in (PROJECT / "src").rglob("*.py"):
        target = code_root / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    for name in ["professor_data_20261030.py", "question_difficulty_inputs.py", "pilot_metric_extensions.py",
                 "run_professor_campaign_20261030.py", "architecture_pilot.py", "architecture_study.py",
                 "verify_professor_release_20261031.py", "package_professor_release_20261031.py"]:
        target = code_root / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(PROJECT / "scripts" / name, target)
    files = {}
    for path in payload.rglob("*"):
        if path.is_file():
            assert not path.is_symlink()
            assert path.suffix not in {".pt", ".npz", ".bundle", ".gz", ".zip"}
            files[path.relative_to(payload).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write(payload / "MANIFEST.json", {"package": NAME, "files": files,
        "source_results_commit": "c286bcbcee46a5317dbb3071c1f6c3a66335d04a",
        "source_handoff_commit": "49982fa7b20c33dc344ef3501b29a03fc35b8bfc",
        "scope": "Portable reporting evidence and checkpoint references; weights and datasets external; no new experiment."})
    staged_check = check(payload)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(payload.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=NAME + "/" + path.relative_to(payload).as_posix())
    with tempfile.TemporaryDirectory(prefix="professor-package-verify-", dir=WORK) as location:
        with zipfile.ZipFile(ARCHIVE) as archive:
            assert archive.testzip() is None
            names = archive.namelist(); assert len(names) == len(set(names))
            for member in archive.infolist():
                path = Path(member.filename)
                assert not path.is_absolute() and ".." not in path.parts
                assert not stat.S_ISLNK(member.external_attr >> 16)
            archive.extractall(location)
        extracted = Path(location) / NAME
        checked = check(extracted)
        assert checked == staged_check
        external = subprocess.run([sys.executable, str(extracted / "verify_package.py")], cwd="/tmp",
                                  text=True, capture_output=True, check=True)
        assert json.loads(external.stdout)["status"] == "PASS"
    record = {"status": "PASS", "archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
              "sha256": sha(ARCHIVE), "payload_check": checked, "crc_readback": "PASS",
              "safe_extraction": "PASS", "standalone_verifier_from_unrelated_cwd": "PASS",
              "original_report_files_unchanged": len(expected_old), "weights_and_raw_data_included": False,
              "remote_push_or_publication": "None"}
    write(OUT / "archive_verification.json", record)
    write(BACKUP / "archive_verification.json", record)
    ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256").write_text(record["sha256"] + "  " + ARCHIVE.name + "\n")
    assert expected_old == {str(p.relative_to(OLD)): sha(p) for p in OLD.rglob("*") if p.is_file()}
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
