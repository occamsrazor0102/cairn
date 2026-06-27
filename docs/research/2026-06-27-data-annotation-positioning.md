# CAIRN repositioning research: data-annotation pipelines and Pangram differentiation

**Date:** 2026-06-27
**Status:** Research input for a PRD revision (not a product commitment)
**Method:** Deep-research fan-out (110 agents): 6 search angles, 27 sources
fetched, 120 candidate claims extracted, 25 adversarially verified (2-of-3
refute kills a claim). Result: 24 confirmed, 1 killed. Every quantitative
figure below is tagged as *verified primary*, *peer-reviewed*,
*self-reported*, or *contested* so the PRD does not overclaim.

---

## Executive summary

LLM contamination of human data-annotation pipelines is an empirically
established, peer-reviewed phenomenon, not a hypothetical. Prevalence
estimates cluster at roughly 30 to 46% of crowdworkers covertly using LLMs
on free-text tasks. Covert use survives policy and UX countermeasures, so
detection (not policy alone) is required, yet the segment is under-tooled:
only about 11% of teams use an automated detector. The harm ties directly
to AI-lab buyer pain: contaminated human data is high-quality but
homogeneous, and training on synthetic data drives "model collapse,"
making genuinely human data scarce and valuable.

Pangram is the strongest named competitor, and it **already markets to this
buyer** (an ML-engineer / RLHF use-case page), but with a black-box cloud
classifier and no productized annotation tooling. That is the opening for
CAIRN: a high-value, under-served annotation buyer where CAIRN's calibrated
probability, conformal uncertainty, honest abstention, per-feature
evidence, local-first privacy, contributor-graph, and audit trail map to
documented needs, while Pangram's genuine strengths (very low self-reported
false-positive rate, funding, brand) must be acknowledged rather than
disputed.

---

## 1. The problem: LLM contamination of annotation pipelines

### Prevalence (verified)

| Evidence | Figure | Source and strength |
|---|---|---|
| MTurk workers using LLMs on an abstractive-summarization task | 33 to 46% | Veselovsky et al. 2023, peer-reviewed CACM 2024 (keystroke + classifier). Verified primary. |
| Follow-up, summarization task, no LLM guidance | ~30% | Veselovsky et al. (CACM 2024). Verified primary. |
| Prolific participants self-reporting LLM use on open-ended questions | 34% | Zhang et al. 2025, peer-reviewed (Sociological Methods and Research). Verified primary. |
| NLP researchers who observed LLM use in their own crowdsourced data | 44% | 2026 community survey (n approx. 100 free-text requesters). Self-report, selection-bias caveat (2-1 vote). |
| Participants copy/pasting on an open-ended page (delegation proxy) | 45% | 2025 pilot. Noisy proxy. |

The literature names the phenomenon "LLM contamination."

### Policy does not solve it; detection is required (verified)

Even when workers are explicitly forbidden, up to 24% still use LLMs. The
best combined intervention ("do not use LLMs" plus disabling copy-paste)
only cut use from 27.6% to 15.9%. Mitigations "significantly reduce, but
not eliminate" covert use (Veselovsky, CACM 2024). This is the load-bearing
argument for a detector rather than a policy.

### The harm maps to AI-lab buyer pain (verified, with one caveat)

Contaminated human data is "high-quality but homogeneous," collapsing the
long tail of human behavior. Training on synthetic or contaminated data
drives "model collapse" (Shumailov et al., Nature 2024, cover article):
"tails of the original content distribution disappear," and "the value of
data collected about genuine human interactions ... will be increasingly
valuable." Related: preference leakage contaminates LLM-as-judge evaluation
(Li et al. 2025).

Honesty guardrail: model-collapse severity is contested. Gerstgrasser et
al. (2024) show that accumulating real and synthetic data together bounds
the error. Motivate detection without claiming collapse is inevitable.

### Scope limit and one refuted claim (must disclose)

Black-box text detectors (the class CAIRN and Pangram both belong to) are
documented as ill-suited to structured or discrete labels (multiple-choice,
preference-pick): "discrete responses lack the rich textual features needed
for classification" (Zhang et al. 2025). CAIRN's text detection plausibly
wins on free-text annotation, SFT demonstrations, and RLHF rationales, but
does not address multiple-choice or preference-pick contamination.

Refuted (0-3, do not assert): the claim that contamination is
"concentrated in subjective RLHF/preference/toxicity tasks." The evidence
refuted it.

## 2. The buyer: AI data pipelines, under-tooled

- Only about 11% of researchers use an automated AI-detection tool. The
  rest rely on informal heuristics: AI-like writing style (63.6%), unusually
  fast completion (63.6%), and high cross-worker similarity (31.8%). The
  last is exactly CAIRN's cross-submission / contributor-graph signal, done
  manually today.
- Prolific ships behavioral "LLM authenticity checks" (15+ patterns,
  self-reported 98.7% precision on free-text), but these are browser and
  keystroke-side (copy-paste, tab-switching), which makes them complementary
  to content-side analysis rather than competing. Prolific's self-reported
  recall is only 78.9%, so it misses a substantial share. Prolific states
  covert ChatGPT use is "a far more common threat than AI agents."
- Buyer needs (synthesized): very low false-positive tolerance, batch and
  API throughput, privacy and IP protection of proprietary prompts, and
  auditability.
- Gap: beyond Prolific, public disclosures from Scale AI, Surge AI, Appen,
  Toloka, Mercor, Invisible, and Labelbox were not verified in this pass,
  and there are no verified market-size dollar figures. Treat vendor-specific
  buyer claims and sizing as research-needed.

## 3. Pangram deep-dive

- **Company:** Pangram Labs, Brooklyn NY; co-founders Bradley Emi (CTO) and
  Max Spero (CEO), ex-Stanford, founded 2023. Funding approximately $4M
  (about $1.25M pre-seed led by Haystack VC, plus about $2.7M seed led by
  ScOp with Script Capital and Cadenza; "$4M seed" per BusinessWire,
  2025-06-23). Company-sourced press.
- **Method (verified from their own technical report):** a transformer
  classifier trained via mirror-prompting (synthetic "mirrors" of specific
  human texts) plus hard-negative mining framed as coreset selection.
- **Claims (self-reported on own or self-run test sets):** about 99%
  accuracy; 0.02% domain-weighted false-positive rate after hard-negative
  mining; roughly 0.00 to 0.05% in most domains (poetry the outlier at
  3.50%); an explicitly engineered 5x false-positive-to-false-negative
  asymmetry; RAID 99.44% accuracy / 0.05% FPR (self-run on an external
  benchmark).
- **Independent signal:** UChicago Becker Friedman Institute (Aug 2025)
  found Pangram's FPR "essentially 0" (the only detector meeting a 0.5% FPR
  cap), but measured 2 to 4% false-negative rates. Critics (Narayanan; an
  independent 1/800 vs claimed 1/10,000) dispute the absolute magnitude.
  Net: Pangram's low-FPR posture is a genuine engineered strength whose
  exact numbers are contested.
- **Pricing:** Free (4 credits/day); Individual $20/mo (600 scans);
  Professional $65/mo (3000); Developer API $25/500 credits (about $0.05 per
  1k words); Team $20/seat; Enterprise custom. The pricing page carries no
  accuracy or FPR numbers and no annotation SKU.
- **Critical nuance for honest differentiation:** Pangram already markets to
  this buyer. Its `/use-cases/ml-engineers` page states verbatim: "Ensure
  your Human Feedback (RLHF) data is actually human. Detect if crowd-workers
  are using ChatGPT to generate responses for your fine-tuning tasks." So
  the opening is not "Pangram ignores annotation." It is that Pangram
  gestures at the buyer with a black-box single classifier and no
  productized annotation tooling (no calibrated uncertainty, no abstention,
  no contributor-graph, no audit chain, no local-first deployment). CAIRN's
  wedge is productizing and differentiating, not virgin whitespace.

## 4. CAIRN vs Pangram (for the data-pipeline buyer)

| Dimension | Pangram | CAIRN | Edge |
|---|---|---|---|
| Core method | Single transformer classifier | Multi-detector fusion + trained calibration | CAIRN (diversity/robustness), unproven head-to-head |
| False-positive rate | 0.02% self-reported, independently corroborated as near-zero | Calibrated, no public ultra-low-FPR claim | Pangram (real strength) |
| Calibrated probability + uncertainty interval | Calibration mentioned; point score | Conformal interval + per-detector calibration | CAIRN |
| Honest abstention (UNCERTAIN) | No | Yes | CAIRN |
| Per-feature evidence / spans | Interpretability highlights | Spans + ranked contributions | Even / CAIRN |
| Privacy / local-first | Cloud API | Loopback / local-first; text stays on machine | CAIRN (IP-sensitive prompts) |
| Batch + relational (repeat/coordinated) | Batch API | Batch + cross-submission similarity + contributor-graph | CAIRN |
| Audit trail | No | Hash-chained provenance | CAIRN |
| Discrete / multiple-choice labels | Ill-suited | Ill-suited | Neither (shared limit) |
| Scale / brand / proven low FPR | Strong, funded, independent corroboration | Earlier-stage | Pangram |

## 5. Competitive map (research baselines, lightly verified this pass)

Research-grade detectors used for dataset filtering include Binoculars
(ICML 2024, zero-shot perplexity-ratio), Fast-DetectGPT, and the
RoBERTa-base OpenAI detector. The academic literature also validates
CAIRN's own thesis: conformal-prediction machine-generated-text detection
with a bounded false-positive rate (ACL 2025). Detailed head-to-head
performance on annotation-length text was not verified in this pass and is
flagged as a gap.

## 6. Recommended PRD changes

1. New customer segment in section 2: "AI data pipelines / labeling and
   RLHF integrity." Buyers: data-labeling vendors, frontier-lab data teams,
   academic and evaluation teams. Privacy pressure: Very High (proprietary
   prompts and IP).
2. New persona: "Mara, the Data-Quality / RLHF Operations Lead," who must
   keep human-feedback free-text actually human, at batch scale, without
   leaking proprietary prompts.
3. New use case in section 3: batch-screen free-text annotations, SFT
   demonstrations, and RLHF rationales for covert LLM origin; use the
   contributor-graph to catch repeat or coordinated synthetic submitters;
   export an auditable integrity record per batch.
4. Competitor-table additions in section 2: a Pangram row (strong low-FPR
   cloud classifier, education-positioned, already gesturing at an RLHF
   use-case, but black-box, cloud-only, no calibrated uncertainty or audit)
   and the research-detector baselines.
5. Differentiation messaging: calibrated, abstaining, evidence-bearing,
   local-first detection productized for the data-pipeline buyer, versus a
   cloud point-classifier.
6. Honesty guardrails: disclose the discrete-label limitation; do not claim
   model collapse is inevitable; mark Pangram's FPR as a real but contested
   strength; do not claim RLHF-task concentration (refuted).

## Caveats and open questions

- All Pangram and Prolific accuracy / FPR / precision numbers are vendor
  self-reported on their own or self-run test sets. The strongest
  independent signal is UChicago (Aug 2025), which corroborates Pangram's
  near-zero FPR directionally but measures 2 to 4% false negatives.
- Prevalence numbers are detection or observation-based lower bounds with
  method-dependent variance; the 44% survey carries self-selection bias.
- Pangram ships fast: this is scoped to Pangram 3.1 (model card 2026-01-16)
  and technical report v3 (2402.14873v3). Later versions exist.
- Not established this pass: specific data-vendor disclosures beyond
  Prolific; market-size dollar figures; any Pangram lab/vendor partnership;
  head-to-head detector benchmarking on annotation-length text with
  calibration and abstention behavior (not just point accuracy/FPR).

## Sources

- Veselovsky et al. 2023 / CACM 2024 (arXiv:2306.07899): https://arxiv.org/abs/2306.07899
- Prevalence and Prevention of LLM Use in Crowd Work: https://www.researchgate.net/publication/389116595_Prevalence_and_Prevention_of_Large_Language_Model_Use_in_Crowd_Work
- Zhang et al. 2025, contaminated crowdsourcing (arXiv:2506.06991): https://arxiv.org/pdf/2506.06991
- 2026 NLP-researcher survey (arXiv:2606.04924): https://arxiv.org/html/2606.04924
- Crowdwork pilot (arXiv:2508.01390): https://arxiv.org/html/2508.01390v2
- Model collapse, Nature 2024: https://www.nature.com/articles/s41586-024-07566-y
- Pangram 3.1 model card: https://www.pangram.com/research/model-card/pangram-3-1
- Pangram technical report (arXiv:2402.14873v3): https://arxiv.org/html/2402.14873v3
- Pangram ML-engineers use-case: https://www.pangram.com/use-cases/ml-engineers
- Pangram pricing: https://www.pangram.com/pricing
- Pangram seed funding, BusinessWire: https://www.businesswire.com/news/home/20250623805035/en/Pangram-Closes-$4-Million-in-Seed-Funding-for-AI-Detection-Technology
- Prolific authenticity checks: https://www.prolific.com/resources/how-prolific-detects-bots-and-ai-in-online-research
- Binoculars (arXiv:2401.12070): https://arxiv.org/abs/2401.12070
- Conformal MGT detection, ACL 2025 (arXiv:2505.05084): https://arxiv.org/abs/2505.05084
- Preference leakage (arXiv:2502.01534): https://arxiv.org/pdf/2502.01534
