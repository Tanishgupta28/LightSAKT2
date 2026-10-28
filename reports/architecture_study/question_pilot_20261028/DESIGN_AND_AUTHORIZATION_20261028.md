# Frozen question-difficulty pilot, 28 October 2026

Current user authorizes one strongest evidence-supported ASSIST2009 seed42 run
under existing shared-GPU permission,4GiB allocator and900second cap,after CPU
correctness and admission. Diagnostics chose known-problem difficulty over a
weaker practice-count candidate. No additional pilot,search,retry or test use.

Variable: add alpha * [logit(problem_rate)-logit(skill_rate)] to original model
logits. One scalar alpha initialized0. Original parameter init/RNG/initial
predictions,causal ELU+1,one block,residuals,norms,FFN and all training controls
unchanged. Problem ID is additional known metadata;this is an input-representation
gain hypothesis,not a same-input kernel superiority claim.

For training rows,ALL labels from the current student are removed from BOTH
question and skill rate estimation. For validation rows,only training labels
are used. Skill rate=(success+1)/(count+2);question rate=(success+10*skill_rate)/
(count+10).10 was fixed in the diagnostic,no smoothing sweep. Unseen questions
fall back to skill rate,giving zero correction. No validation label is used in
features. No features/test dataset for the final test partition are constructed.
The adapter changes only the dedicated process's dataset binding/loss batch view,
restoring both on exit;original training engine source is unchanged. Six-field
batches keep valid last;epoch reporting receives original five-field view.

All original splits,sequence100,d64,2heads,dropout0.1,batch64,Adam lr0.001,
betas(0.9,0.999),epsilon1e-8,weight decay0,float32/no AMP/TF32,50epoch/patience5,
strict validation-AUC selection retained. Dataset/protocol/feature/config hashes
frozen. Same student-disjoint evaluation but richer known item information than
historical models;future matched-input controls are needed for model comparisons.

General holds stay unchanged;only an expiring exact-config/hold exception permits
this question stage. Existing10GiB admission/6GiB runtime-free,4GiB RSS/2GiB host
headroom,0.1s pacing,30s step-stop and unsupported-MIG cap/headroom fallback apply.
No exclusivity claim,no peer signalling;operator window is not a scheduler lease.
