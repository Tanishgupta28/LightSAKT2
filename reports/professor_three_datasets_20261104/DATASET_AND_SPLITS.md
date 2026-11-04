# Replacement dataset and split specification

EdNet-KT1 was selected because its actual publisher-linked payload supplies
authentic `question_id`, student sequences and official question answers/tags.
The processed benchmark is **EdNetKT1_5k_first**, a deterministic 5,000-student,
first-question-per-solving-session subset. Always use that qualification.

| Criterion | EdNet-KT1 | Junyi alternative |
|---|---|---|
| Stable repeated item identity | Authentic question_id verified against 13,169 official question records | Individual generated-question identity not verified; exercise/template/counter fields cannot simply be assumed question IDs |
| Student sequences | 784,309 authentic user files in the obtained release | Student histories described in original study; matching original payload not obtained here |
| Completeness/integrity | Exact publisher-linked archives downloaded; selected source files CRC/row/label checked | Historical DataShop endpoint returned403; no equivalent payload integrity check performed |
| Existing feature fit | Same item-vs-skill correction directly applicable | Exercise-level correction would require a differently labelled protocol unless a real item crosswalk is obtained |
| Reproducibility | Fixed filenames, seeds, source/split hashes, authoritative answers and recorded filtering | Release/schema/item-identity verification remains necessary |
| Storage/compute | 1.2GB compressed raw KT1; deterministic 5k subset; guarded sequential native runs feasible | Not benchmarked; no memory/runtime figures invented |
| Research interpretation | Known item-level offline experiment, explicitly qualified reduced population | Potential exercise-level study, not automatically the same question-level claim |

EdNet provides the strongest actually verified fit to the approved modification.
Its use avoids relabelling categories or synthetic IDs and fits the shared-resource
budget. The complete archive is retained; subset selection was fixed without
labels or model outcomes. A broader full-KT1 study would require a new budget.

## Authoritative files and integrity

Publisher: https://github.com/riiid/ednet

Paper: https://arxiv.org/abs/1912.03072

The publisher links `https://bit.ly/ednet_kt1` to the public Drive file
`1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw`:
https://drive.google.com/file/d/1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw/view

Its Contents link `https://bit.ly/ednet-content` resolves to
https://drive.google.com/file/d/117aYJAWG3GU48suS66NPaB82HwFj6xWS/view

| Payload | Bytes | Locally computed SHA256 |
|---|---:|---|
| EdNet-KT1.zip | 1201163816 | 0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535 |
| EdNet-Contents.zip | 173976 | aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7 |
| contents/questions.csv | 554286 | 137d2936b746c98e1586ef134bf9f9bc4b5a878c96b88e630b2241fe9f15d3cd |

These are fingerprints of actually downloaded public files, not published
provider checksums. Download used the official public file after the Drive
connector's 256MiB response-size limit rejected the 1.2GB file. No credentials
or transient download URLs belong in the reporting archive.

The ZIP contains 784,309 distinct authentic user CSVs and one directory entry;
user content totals 3,072,366,053 uncompressed bytes. All 5,000 selected files
were completely read and passed ZIP CRC. The unselected files were not all
decompressed or CRC-checked. Questions metadata has 13,169 unique authentic IDs,
official answers, parts1–7, bundle IDs and tags. Select the exact archive member
`contents/questions.csv`, excluding `__MACOSX` resource forks.

KT1 columns are `timestamp,solving_id,question_id,user_answer,elapsed_time`.
Metadata columns are `question_id,bundle_id,explanation_id,correct_answer,part,tags,deployed_at`.
Official answer keys and raw answers construct labels only; elapsed time,
deployment date, explanation ID and correct answer are not model predictors.

## Population and chronological events

1. Sort all authentic user filename stems lexicographically. Draw 5,000 without
   replacement with `numpy.random.default_rng(20261104).choice`; sort the selected
   filenames for processing. No labels, lengths or model scores affect selection.
2. Keep original student and question identifiers. Validate numeric nonnegative
   integer timestamps/session IDs; keep source-row positions for stable ties.
3. Remove exact duplicates of the five original KT1 fields. Sort each student by
   timestamp and source row. Fail closed if two solving sessions' timestamp
   ranges overlap or tie. All selected files passed.
4. Choose the FIRST presented question of each solving session before checking
   answer validity. Exclude the other session questions from both target and
   history. This avoids exposing another current-bundle answer through history.
5. Require a matching authentic metadata question. Keep valid answers a–d,
   except part2 permits a–c. Incorrect but valid answers remain labelled0.
   An invalid first answer is excluded, never replaced by another session item.
6. Correctness is recorded answer == official correct answer. Retain repeated
   authentic questions in different sessions; no question-based deduplication.
7. Full sorted unique metadata tagsets are composite skill categories; do not
   explode multiskill rows or replace item IDs with tags. Missing tags have an
   explicit category; no missing tags occurred in the official metadata.

Counts: 721,760 raw selected rows, 34 exact duplicates, 563,565 sessions,
158,161 other within-session questions excluded, zero metadata misses and
96 invalid first answers excluded. Final: 563,469 interactions, 5,000 students,
11,893 authentic items. There are 4,816 students with at least two observations;
single-observation students remain selected but have zero next-response targets.

This is a fixed resource-feasible subset, not full KT1 or an official leaderboard
protocol. First-position selection changes the item population and may introduce
position bias. Presentation/session ordering is explicit; wall-clock availability
of all historical responses in an online deployment is not independently proven.

## Student split, encoding and targets

The original `src.datasets.preprocess.split_students` function is reused with
seed42 and 70/10/20 student proportions. Split sets are disjoint. Vocabulary
contains training tagsets only; UNK0 is a valid unknown skill distinguished from
padding by explicit validity masks.

| Split | Students | Interactions | Next-response targets | Unknown-skill rows |
|---|---:|---:|---:|---:|
| Train | 3500 | 407242 | 403742 | 0 |
| Validation | 500 | 43375 | 42875 | 1 |
| Test | 1000 | 112852 | 111852 | 4 |

Sequences use native length100 chunks, stride100, one preceding interaction at
each boundary and every n−1 target once. Current query skills/items are known;
current responses and future responses are excluded. Full target-array equality
to each student's labels[1:] is checked independently after checkpoint replay.

Processed CSV SHA256:
`e31e9e9cefd36d2d671ff7d3e3baad8355a97afc7e3996eb987fc09c2444e25a`

Student split SHA256:
`a31a01019f204e314e90ee3b192f1a1b5d4d02b348f405ceb1a75bc94972ef9a`

Selected population SHA256:
`14815bfd9992e4d05878e3b8d882b2106bd183503791b1b68db60dc07c5880b7`

Full selected student IDs, individual source member CRC/counts and exact split
IDs are preserved locally, backed up with the new experiment artifacts and
included as population/split manifests in the reporting archive. No raw answers
or per-student prediction/target arrays are shipped. Reconstruction from the
publisher archive and fixed algorithm recovers the population with these hashes.

## Difficulty fitting and self-label safeguards

Use the unchanged question-vs-skill logit correction with item smoothing10 and
skill Beta(1,1). Exclude the entire current student from both counts on training
examples. Validation/test queries are transformed against training rows only,
without passing their outcome labels to the transform. Unseen items yield0.
Training item aggregates cover 11,787 real questions. Held-out unseen-item row
fractions: validation0.000668588, test0.001648176.

Own-student label flips, held-out label flips, independent feature recomputation
and a responsive other-student positive control passed. No within-student self
label, held-out target or future held-out label enters the correction. Rates
still use the complete offline training corpus, including later observations of
OTHER training students. This is not a chronological deployment experiment.

## Alternative and replaced release

Junyi was examined but not selected. The original author paper distinguishes
exercises containing multiple problems; individual generated-question identity
was not verified from examined exercise/problem-type metadata. A practice
counter is not a stable item ID. Exercise-level difficulty would require a
distinctly labelled protocol. This conclusion does not exhaust all Junyi releases.

ASSIST2015 remains documented in the preserved October31 release: four columns,
no authentic independent item identifier/crosswalk recovered. Historical baseline
results remain unchanged; its difficulty results are unavailable and are not
presented as completed three-dataset evidence.
