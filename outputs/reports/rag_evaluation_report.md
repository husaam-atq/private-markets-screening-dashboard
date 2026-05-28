# RAG Evaluation Report

This report evaluates whether the filing retrieval layer returns source-backed evidence for diligence-style questions. The gold set includes section-confuser and no-answer questions, so perfect-looking metrics should not be assumed in live refreshes.

## Summary Metrics

| Metric | Value |
|---|---:|
| questions_evaluated | 96.0 |
| answerable_questions | 72.0 |
| hit_rate_at_3 | 100.0% |
| hit_rate_at_5 | 100.0% |
| precision_at_3 | 86.6% |
| precision_at_5 | 82.5% |
| mrr | 99.3% |
| ndcg_at_5 | 96.8% |
| section_match_rate | 95.8% |
| chunk_level_hit_at_5 | 100.0% |
| citation_coverage | 100.0% |
| unsupported_claim_rate | 0.0% |
| no_answer_accuracy | 100.0% |
| evidence_strength_score | 12.9% |
| numeric_consistency_checks | 100.0% |

## Strong and Weak Retrieval Examples

- strong_retrieval: MSFT MSFT_revenue Precision@5 1.00, top score 0.147, top source sample_fallback.
- strong_retrieval: MSFT MSFT_customer Precision@5 1.00, top score 0.194, top source sample_fallback.
- strong_retrieval: MSFT MSFT_liquidity_debt Precision@5 0.80, top score 0.091, top source sample_fallback.
- strong_retrieval: MSFT MSFT_margin_confuser Precision@5 0.60, top score 0.144, top source sample_fallback.
- strong_retrieval: MSFT MSFT_capex_allocation Precision@5 0.80, top score 0.114, top source sample_fallback.
- strong_retrieval: MSFT MSFT_competition_regulation Precision@5 0.80, top score 0.185, top source sample_fallback.
- strong_retrieval: HCA HCA_revenue Precision@5 0.80, top score 0.071, top source sample_fallback.
- strong_retrieval: HCA HCA_customer Precision@5 0.80, top score 0.160, top source sample_fallback.
- weak_retrieval: SOFI SOFI_customer Precision@5 0.20, top score 0.084, top source real_sec_filing.

## Missed Answerable Questions

No answerable gold questions missed at top 5 in the current evaluation, but Precision@5 still shows that some retrieved chunks are adjacent rather than directly responsive.