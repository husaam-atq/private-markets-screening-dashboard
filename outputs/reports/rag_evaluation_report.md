# RAG Evaluation Report

This report evaluates whether the filing retrieval layer returns source-backed evidence for diligence-style questions.

## Summary Metrics

| Metric | Value |
|---|---:|
| questions_evaluated | 70.0 |
| answerable_questions | 60.0 |
| hit_rate_at_3 | 100.0% |
| hit_rate_at_5 | 100.0% |
| precision_at_3 | 78.9% |
| precision_at_5 | 71.3% |
| mrr | 100.0% |
| ndcg_at_5 | 95.9% |
| section_match_rate | 100.0% |
| citation_coverage | 100.0% |
| unsupported_claim_rate | 0.0% |
| no_answer_accuracy | 100.0% |
| evidence_strength_score | 24.4% |
| numeric_consistency_checks | 100.0% |

## Weak Retrieval Cases

No answerable gold questions missed at top 5 in the cached sample evaluation.