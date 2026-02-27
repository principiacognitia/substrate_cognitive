# Substrate-Independent Cognitive Architecture: T-maze MVP

## Статус: MVP v5.0 завершён (Qualitative GO)

Эта репозитория содержит минимальную реализацию когнитивной архитектуры на основе:
- **Active Inference** (belief state over hidden rules)
- **Reology** (viscosity-based habit formation: V_G, V_p)
- **Multi-Gate** (MF/MB arbitration based on uncertainty)

## Ключевые результаты

| Компонент | Статус | Примечание |
| :--- | :--- | :--- |
| Belief State (b_t) | ✅ Работает | Сходится за 10-15 триалов |
| Rheology (V_G, V_p) | ✅ Работает | Рост в стабильности, плавление при ошибке |
| Gate Switching | ⚠️ Частично | EXPLORE фаза короткая (1-3 триала) |
| Hysteresis | ⚠️ Частично | V_G задержка 5-10 триалов после changepoint |

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│  Environment (T-maze, changepoint @ trial 60)               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Agent                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Belief (b)  │  │ Q_MF / Q_MB │  │ Rheology (V_G, V_p) │  │
│  │ 2 hypotheses│  │ Action vals │  │ Viscosity update    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Gate (MF/MB arbitration)                               ││
│  │  u_t = [u_delta, u_s, u_v, u_c]                         ││
│  │  EXPLORE if sigmoid(w·u) * (1-V_G) > theta_MB           ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Запуск

```bash
python run_experiment.py
```

## Структура

| Файл | Описание |
| :--- | :--- |
| config.py | Все гиперпараметры (единый источник истины) |
| belief.py | Belief state (Dirichlet-like update over 2 hypotheses) |
| gate.py | Gate selection + rheology update |
| agent.py | Agent state (Q, b, V_G, V_p) |
| environment.py | T-maze environment with changepoint |
| metrics.py | Go/No-Go criteria evaluation |
| run_experiment.py | Main experiment loop |
| visualize.py | 4-panel dashboard (accuracy, mode, V_G, V_p) |

## Выводы MVP

1. **Reology работает:** V_G растёт в стабильной среде и плавится при структурных изменениях.
2. **Belief state необходим:** Без u_s (belief entropy) гейт не переключается на EXPLORE.
3. **T-maze слишком прост:** Для устойчивой EXPLORE фазы нужна задача сложнее (Two-Step, multi-arm).
4. **Параметры v5.0:** BELIEF_LR=0.12, ALPHA_MF=0.12, u_s_weight=3.5 — хороший базис для будущих экспериментов.

## Следующие шаги

- [ ] Two-Step Task (Daw et al., 2011) для валидации MB/MF арбитража
- [ ] Добавить episodic buffer + prioritized replay
- [ ] Сравнение с биологическими данными (крысы, T-maze VTE)
- [ ] Абляции (No Gate Rheology, No Belief, MF-only)

## Лицензия

MIT
