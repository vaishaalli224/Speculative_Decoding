# Numbers behind the figures (table-view twin)

## Fig 1 — stage-wise τ (pooled [bootstrap 95% CI])

| Draft | xLAM τ | TB τ |
|---|---|---|
| Untuned Qwen2.5-Coder-0.5B | 3.32 [3.23–3.43] | 2.63 [2.54–2.77] |
| n-gram (no model) | 0.72 [0.61–0.79] | 0.58 [0.49–0.64] |
| Stage-1 KD (xLAM ctxs) | 4.01 [4.17–4.40] | 2.65 [2.61–2.84] |
| TB-KD (TB ctxs) | 4.07 [4.19–4.42] | 3.04 [3.01–3.26] |
| Stage-2 GKD 1:1 | 4.18 [4.34–4.54] | 2.94 [2.97–3.23] |
| Stage-2 GKD TB-only | 4.05 [4.20–4.43] | 3.04 [3.02–3.28] |

## Fig 2 — wall-clock (median of 3, greedy, k=5)

| Panel | Config | tok/s | speedup |
|---|---|---|---|
| xLAM b1 | AR | 66.7 | 1.00× |
| xLAM b1 | n-gram | 111.5 | 1.67× |
| xLAM b1 | Untuned | 148.0 | 2.22× |
| xLAM b1 | Stage-1 KD | 169.3 | 2.54× |
| xLAM b1 | GKD TB-only | 163.8 | 2.45× |
| xLAM b32 | AR | 582.1 | 1.00× |
| xLAM b32 | n-gram | 899.8 | 1.55× |
| xLAM b32 | GKD TB-only | 1365.8 | 2.35× |
| TB per-turn | AR | 66.7 | 1.00× |
| TB per-turn | n-gram | 101.6 | 1.52× |
| TB per-turn | GKD TB-only | 141.2 | 2.12× |

## Fig 3 — α by region (codes: 1 prose / 2 JSON / 3 tags / 10 name / 11 args / 4 final answer)

| Draft | Eval | assistant prose | tool-call JSON | wrapper tags | call name | call args | final answer | region 0 (not drawn) |
|---|---|---|---|---|---|---|---|
| Untuned 0.5B | xLAM | 0.935 | 0.875 | 0.623 | 0.785 | 0.931 | — | 0.962 (n=79) |
| Untuned 0.5B | TB | 0.810 | 0.820 | 0.832 | 0.802 | 0.835 | 0.801 | 0.817 (n=5090) |
| Stage-1 KD | xLAM | 0.941 | 0.948 | 0.878 | 0.934 | 0.956 | — | 0.948 (n=77) |
| Stage-1 KD | TB | 0.815 | 0.821 | 0.865 | 0.801 | 0.837 | 0.802 | 0.799 (n=4937) |
| TB-KD | xLAM | 0.947 | 0.952 | 0.889 | 0.946 | 0.956 | — | 0.987 (n=79) |
| TB-KD | TB | 0.854 | 0.857 | 0.874 | 0.846 | 0.866 | 0.832 | 0.853 (n=4913) |

## Fig 4 — per-position αₙ

| Draft | Eval | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 |
|---|---|---|---|---|---|---|
| Untuned | xLAM | 0.832 | 0.879 | 0.869 | 0.959 | 0.958 |
| Untuned | TB | 0.754 | 0.795 | 0.830 | 0.862 | 0.889 |
| Stage-1 KD | xLAM | 0.914 | 0.947 | 0.948 | 0.945 | 0.965 |
| Stage-1 KD | TB | 0.760 | 0.787 | 0.842 | 0.857 | 0.887 |
| TB-KD | xLAM | 0.911 | 0.959 | 0.945 | 0.966 | 0.970 |
| TB-KD | TB | 0.806 | 0.854 | 0.859 | 0.884 | 0.896 |
