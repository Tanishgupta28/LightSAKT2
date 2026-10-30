# Frozen comparison and explicit test release — 30 October 2026

Current user Phase3 authorizes missing multi-seed experiments where existing
shared permission and admission checks pass. Phase4/5 explicitly request final
test measurements and replay. All9new training configs and dataset feature
semantics were frozen before evaluating any new modified-model test outcome.
No new loss,attention,depth,hyperparameter search or architecture is introduced.
Thirty baseline seeds and modifiedASSIST2009seed42 are reused. Training itself
has evaluate_test=false;test replay is a separate CPU-only phase after checkpoint
selection. Test scores cannot affect configs,seeds,stopping or model selection.

The same one-scalar correction is used. ASSIST2009 retains exact old numerical
training/validation features. Algebra2005 uses target tutor-step identity from
(Problem Name,Step Name),mapped exactly to originalprocessedsource_row/student/
FirstTransactionTime/skill. All original splits/data remain unchanged. ASSIST2015
currentfour-columnrelease has no separateitemidentity;that row stays PENDING.

Existing intentionalsharedMIG permission,exactapprovedUUID and finite admission
are verified. PeerhostPID ownership unresolved;no exclusivity claim,peer signals
or ownership assumptions. Owned worker PID/runidentity is logged in its observer.
Same4GiB allocator,900s perrun,10GiB admission/6GiB runtimefree,4GiB RSS/2GiB host
headroom,0.1s pacing,30s stepstop,unsupportedMIGfallback. Nine new sequential
windows,<=8100GPUseconds,total135minuteceiling. UserPhase3 scope applies now;
old29October preparation-only scope does not require a redundant confirmation.
Stop on any unsafe condition,no automatic retry or extra budget. General hold
bytes and prior artifacts remain protected by hashes;no background scheduler.

Baselines named 'original' mean the established corrected causal reproducible
implementations,not the leaky historical original_snapshot results. The historical
snapshot remains preserved and is not methodologically comparable.
