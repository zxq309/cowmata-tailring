# COWMATA metric contract

Five layers. Everything is reported internally; only layer 5 is quoted to a customer.

## Layer 1 — frame level
Per-class precision / recall / F1, macro-F1, and MoF (mean-over-frames accuracy).
Diagnostic only. Never an acceptance criterion: one cow produces ~172,800 frames
a day and only about ten urinations, so predicting "nothing" scores 99.99%.

## Layer 2 — segment level *(new in 20260819)*
- `F1@{10,25,50}` — one-to-one matching by temporal IoU.
- `edit score` — Levenshtein distance over run labels, in `[0,100]`.
- The historical 2.5 s-tolerance matching is retained and reported alongside.

A fixed 2.5 s tolerance is lenient for a 1 s stand-up and strict for a 35 s
defecation, so two events' numbers are not on the same scale and no external
reviewer recognises the protocol. `F1@k` and `edit score` are what the temporal
action-segmentation literature reports; the edit score in particular quantifies
over-segmentation, which is the diagnosed cause of the low event precision.

## Layer 3 — deployment level
- false alarms per cow per 24 h
- miss rate
- onset localisation error: median and P90 of `|Δt_start|`

## Layer 4 — generalisation *(mandatory)*
Every metric is reported **per cow** as well as pooled, and pooled figures carry
a bootstrap confidence interval obtained by resampling **cows**, not rows.

In the 20260818 dataset one animal held 71.6% of all supervised samples, so a
pooled cross-cow number was effectively that one animal's score. Resampling rows
would have treated its 246,487 samples as 246,487 independent observations.

## Layer 5 — day level (oestrus / calving)
Sensitivity, specificity, PPV, false alerts per cow per week, and the median and
quartiles of **lead time**. An alert that arrives after the fact has a
sensitivity of 1 and a value of 0.

## Evidence gating
An event may not have precision or F1 reported at all unless it has at least
**10 true events across at least 3 cows**, *and* trusted negatives from
exhaustive video review. Otherwise the report says `not_evaluable`. This is
enforced in `metrics.event_level_metrics`, not left to the reader.

`review_coverage` is what makes negatives trustworthy. With no coverage rows the
mask degrades to "intervals that happen to carry a posture label", evaluation
happens only on event-dense stretches, `labelled_hours` is far below the true
value, false-alarm rates come out systematically low and precision systematically
high. **Until exhaustive review exists, precision and false-alarm rates are not
claimable.**

## What to tell a customer
Three numbers, always together:

> On N cows and M hours of independently video-reviewed recording, detection
> rate for standing up / lying down / urination was **X%**, at an average of
> fewer than **Y false alerts per cow per day**.

Never quote an F1 (a customer cannot interpret it) and never quote an accuracy
without a false-alarm rate (it is meaningless at this event frequency).
