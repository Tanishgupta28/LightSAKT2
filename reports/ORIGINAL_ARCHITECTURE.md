# Protected LightSAKT architecture

Authority: `original_snapshot/train/models/light_sakt.py` fetched from the supplied Drive folder. The adjacent historical local copy differs; it is not the source authority. All historical scores and checkpoints are unverified.

For Q identifiers, embedding width d, heads h and capacity L, parameters are M(2Q,d), E(Q,d), P(L,d); learned biased Q/K/V/output projections (d,d); two layer norms (d); FFN d -> floor(d/2) -> d with ReLU and dropout; biased prediction d -> 1. No additional trainable parameters are permitted in the corrected default.

Encoding is x=q+Q*r. M(x)+P supplies keys and values. E(q_next) supplies queries. The attention residual is LayerNorm(dropout(attention)+M(x)+P+E(q_next)). The FFN residual is LayerNorm(FFN(S)+S). Sigmoid predicts r_next.

Corrected head output at aligned index t is phi(Q_t) dot sum_{j<=t,valid} phi(K_j) outer V_j divided by (phi(Q_t) dot sum_{j<=t,valid} phi(K_j)+1e-8), phi=ELU+1. Index t represents history interaction t and target question t+1. Target response t+1 must not contribute to this prediction. Both prefix boundaries include the same history.

Original defects: global numerator uses future values; denominator does not contract query with key sum and combines causal rows; visualization swaps in softmax and changes prediction; no key-padding mask; positional tensor is not sliced for variable length; CPU mask fails on CUDA.

Correction preserves every parameter name and shape. Prefix accumulation is linear in L at fixed d,h, with O(B*L*h*(d/h)^2) materialized training intermediates. This is not constant-memory training. Visualization alone may allocate L^2 weights.

Ablations: no_attention retains the local M+P+E residual pathway and FFN; no_position omits P; single_head changes h only; no_dropout sets dropout to zero. No_regularization is undefined as a distinct historical LightSAKT intervention: dropout is the only explicit default regularizer, Adam weight decay is zero, and DKT+ penalties do not belong to LightSAKT. By user instruction it is reported as equivalent to no_dropout, reusing the same evidence rather than an independent intervention.
