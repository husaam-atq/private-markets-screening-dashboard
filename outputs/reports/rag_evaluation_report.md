# RAG Evaluation Report

This report evaluates whether the filing retrieval layer returns source-backed evidence for diligence-style questions. The gold set includes section-confuser and no-answer questions, so perfect-looking metrics should not be assumed in live refreshes.
Retrieval metrics are separate from optional local language-model generation metrics.

## Summary Metrics

| Metric | Value |
|---|---:|
| questions_evaluated | 96.0 |
| answerable_questions | 72.0 |
| hit_rate_at_3 | 100.0% |
| hit_rate_at_5 | 100.0% |
| precision_at_3 | 98.1% |
| precision_at_5 | 89.4% |
| mrr | 100.0% |
| ndcg_at_5 | 99.5% |
| section_match_rate | 100.0% |
| chunk_level_hit_at_5 | 100.0% |
| citation_coverage | 100.0% |
| unsupported_claim_rate | 0.0% |
| no_answer_accuracy | 100.0% |
| evidence_strength_score | 19.5% |
| numeric_consistency_checks | 100.0% |

## Strong and Weak Retrieval Examples

- strong_retrieval: MSFT MSFT_revenue Precision@5 1.00, top score 0.229, top source sample_fallback.
- strong_retrieval: MSFT MSFT_customer Precision@5 1.00, top score 0.164, top source sample_fallback.
- strong_retrieval: MSFT MSFT_liquidity_debt Precision@5 1.00, top score 0.184, top source sample_fallback.
- strong_retrieval: MSFT MSFT_margin_confuser Precision@5 0.60, top score 0.221, top source sample_fallback.
- strong_retrieval: MSFT MSFT_capex_allocation Precision@5 0.80, top score 0.194, top source sample_fallback.
- strong_retrieval: MSFT MSFT_competition_regulation Precision@5 1.00, top score 0.260, top source sample_fallback.
- strong_retrieval: HCA HCA_revenue Precision@5 0.80, top score 0.151, top source sample_fallback.
- strong_retrieval: HCA HCA_customer Precision@5 0.80, top score 0.138, top source sample_fallback.

## Missed Answerable Questions

No answerable gold questions missed at top 5 in the current evaluation, but Precision@5 still shows that some retrieved chunks are adjacent rather than directly responsive.

## Grounded Generation Context

- answers_evaluated: 80.000
- citation_coverage: 1.000
- unsupported_claim_rate: 0.000
- evidence_overlap_score: 0.774
- banned_language_count: 0.000
- no_answer_accuracy: 1.000
- numeric_consistency_pass_rate: 1.000
- groundedness_score: 1.000

## Limitations and Roadmap

- Hit@5 measures whether at least one relevant passage appears; Precision@5 is more important for user-facing evidence quality.
- Section-confuser questions can retrieve adjacent sections where the filing discusses the same economic issue in different wording.
- Real filing coverage is intentionally capped for commit-safe outputs and can be expanded locally.
- The retrieval benchmark is not a substitute for legal, financial or investment diligence.