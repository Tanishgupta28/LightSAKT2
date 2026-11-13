# Professor-ready progress summary

We verified the original 25-page paper and generated updated tables and figures exclusively from genuine experiments. Across five matched seeds, adding question difficulty improved held-out test AUC from 0.751056 to 0.779359 on ASSIST2009, from 0.810590 to 0.836871 on Algebra2005, and from 0.642417 to 0.706876 on the exact EdNet-KT1 5,000-student first-question-per-session subset. Test accuracy and F1 also improved in every matched seed against original LightSAKT and SAKT.

We reused all 75 existing seed results and evaluated two previously selected ASSIST2009 GKT checkpoints on CPU. Fifteen of the 27 requested five-seed groups are complete; 58 seed results remain missing. No new training was launched because current GPU allocation ownership and a new total campaign budget have not been confirmed.

The gains support adding item information under the documented offline protocol. They do not establish attention-architecture superiority. The feature-matched SAKT + Difficulty control, independent-cohort replication and chronological evaluation remain necessary. Exact five-seed Wilcoxon tests with Holm correction do not establish significance at 0.05. The package is suitable for reviewing verified interim findings, with pending results clearly marked; it is not a complete nine-model paper comparison. The original manuscript remains unchanged.
