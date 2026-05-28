# Grounded Generation Evaluation Report

This report evaluates generated filing answers for citation coverage, evidence overlap, banned recommendation language and unsupported-claim warnings. The default project path is deterministic; local Ollama generation is optional.

## Summary Metrics

| Metric | Value |
|---|---:|
| answers_evaluated | 80 |
| citation_coverage | 100.0% |
| unsupported_claim_rate | 0.0% |
| evidence_overlap_score | 77.4% |
| banned_language_count | 0 |
| no_answer_accuracy | 100.0% |
| numeric_consistency_pass_rate | 100.0% |
| groundedness_score | 100.0% |

## Generation Modes

- deterministic: 80 answers

## Limitations

- Evidence overlap is a practical lexical proxy, not a legal or diligence completeness test.
- Local model output, when enabled, can vary by model and should be reviewed against source filings.
- The guardrails are designed to flag unsupported or recommendation-like wording, not to replace human review.