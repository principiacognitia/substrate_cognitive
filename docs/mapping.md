# Mapping: Latent Variables → Behavioral Invariants → Rodent Proxies

**Version:** 0.2  
**Date:** 26 February 2026  
**Status:** Ready for Stage-2

## 1. Core Correspondence Table

| Latent Variable | Behavioral Invariant in Model                  | Prerequisite          | Rodent Proxy (Observable)                          | Quantitative Metric                                      | Expected Direction          | Falsification Criterion                                      |
|-----------------|------------------------------------------------|-----------------------|----------------------------------------------------|----------------------------------------------------------|-----------------------------|-------------------------------------------------------------|
| V_G             | Hysteresis of control-mode switching           | —                     | Delayed onset of MB-signature after block change   | Latency (trials) to first significant MB-interaction     | Latency(V_G-on) > Latency(V_G-off) | ΔLatency ≤ 0 (ablation contrast, 95% CI)                    |
| V_p             | Perseveration / stickiness                     | —                     | Repetition of previous choice despite feedback     | Stickiness coefficient in logistic regression            | Positive correlation        | No perseveration when V_p high                              |
| u^(s)           | Uncertainty about hidden rule                  | —                     | Elevated VTE / head-scanning at choice-point       | Spearman ρ (u^(s), pause duration)                       | ρ > 0                       | ρ ≤ 0 (bootstrap CI)                                        |
| u^(v)           | Detection of regime shift                      | —                     | Abrupt increase in exploratory behavior            | Changepoint detection in behavior                        | Positive                    | No behavioral reaction to u^(v) rise                        |
| EXPLORE mode    | Active deliberation / MB planning              | —                     | Elevated VTE probability                           | % trials with VTE above threshold                        | EXPLORE > EXPLOIT           | No association with VTE                                     |
| EXPLOIT mode    | Procedural automation                          | —                     | Straight, stereotyped trajectories                 | Path efficiency / low trajectory entropy                 | EXPLOIT < EXPLORE           | No reduction of VTE when u_t low                            |

**VTE / pause / trajectory entropy** — **Spatial-wrapper only** (Akam et al., 2018; Redish, 2016). В чисто абстрактном two-step эти прокси недоступны.

## 2. MB/MF Signature Validation (обязательно для всех прогонов)
- Logistic regression: Stay ~ Reward × Transition  
- MB-signature = interaction coefficient > 0 и 95% bootstrap CI не пересекает 0  
- Проверяется на MF-only и MB-only baseline-симуляциях той же версии задачи (Feher da Silva & Hare, 2018).

## 3. Falsification Criteria (Stage 2)
Модель считается фальсифицированной на текущем этапе, если выполняется хотя бы одно:
1. V_G не производит значимый hysteresis (ΔLatency ≤ 0 по ablation contrast).  
2. u^(s) не коррелирует положительно с VTE-прокси (ρ ≤ 0).  
3. Полный RheologicalAgent не улучшает out-of-sample CV-log-likelihood относительно MF-only / MB-only baselines.  
4. Реология не даёт воспроизводимой диссоциации V_G (control inertia) vs V_p (action inertia) по seeds.

## 4. References (APA 7)
Akam, T., et al. (2018). Deliberation and procedural automation on a two-step task for rats. *Frontiers in Integrative Neuroscience, 12*, Article 30.

Daw, N. D., et al. (2011). Model-based influences on humans' choices and striatal prediction errors. *Neuron, 69*(6), 1204–1215.

Feher da Silva, C., & Hare, T. A. (2018). A note on the analysis of two-stage task results... *PLOS ONE, 13*(4), Article e0195328.

Redish, A. D. (2016). Vicarious trial and error. *Nature Reviews Neuroscience, 17*(3), 147–159.

Wilson, R. C., & Collins, A. G. E. (2019). Ten simple rules for the computational modeling of behavioral data. *eLife, 8*, Article e49547.