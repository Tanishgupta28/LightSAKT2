# Portable LightSAKT results package

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
