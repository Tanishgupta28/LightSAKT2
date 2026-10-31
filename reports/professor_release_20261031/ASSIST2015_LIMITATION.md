# ASSIST2015 question-identifier recovery: unavailable under current protocol

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
