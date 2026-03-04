# Two-Step Task (Daw et al., 2011)

**Статус:** ✅ Завершено (Stage 2A)

## 📖 Описание

Каноническая задача для разделения модель-свободного (MF) и модель-основанного (MB) поведения. Агент делает два последовательных выбора, где переходы между этапами стохастичны (70% common, 30% rare).

### Структура триала

```
Trial Start
    │
    ▼
Stage 1: s1 → a1 (Left/Right)
    │
    ├── Common (70%) ──→ s2 = expected
    │
    └── Rare (30%) ────→ s2 = unexpected
            │
            ▼
Stage 2: s2 → a2 (Left/Right)
            │
            ▼
        Reward (drifting probability)
```

## 🗂️ Файлы

| Файл | Описание |
| :--- | :--- |
| `config_twostep.py` | Все гиперпараметры задачи |
| `env_twostep.py` | Среда (дрейф наград, changepoint) |
| `agent_twostep.py` | RheologicalAgent для Two-Step |
| `debug_interface.py` | Level 0: Interface Contract Test |
| `debug_env_twostep.py` | Level 1: Environment Unit Test |
| `debug_sanity_check.py` | Level 2: MB/MF Signature Validation |
| `debug_integration.py` | Level 3: V_G Dynamics & Hysteresis |
| `debug_ablation.py` | Level 4: V_G Ablation Study |

## 🚀 Запуск

```bash
# Все тесты по порядку
python -m stage2.twostep.debug_interface
python -m stage2.twostep.debug_env_twostep
python -m stage2.twostep.debug_sanity_check
python -m stage2.twostep.debug_integration
python -m stage2.twostep.debug_ablation
```

## 📊 Критерии PASS

| Level | Критерий | Порог |
| :--- | :--- | :--- |
| **0** | Все импорты работают | ✅ No errors |
| **1** | Common transition ratio | 0.65 – 0.75 |
| **2** | MB interaction p-value | < 0.01 |
| **2** | MF interaction p-value | > 0.10 |
| **3** | EXPLORE до changepoint | < 20% |
| **3** | EXPLORE после changepoint | > 50% (в шоке) |
| **4** | Latency Full vs NoVG | p < 0.001 (Mann-Whitney) |

## 📈 Ожидаемые результаты (из логов)

```
Level 2: Sanity Check
  MB Agent: interaction coef = 0.47, p = 0.0000
  MF Agent: interaction coef = 0.09, p = 0.3893

Level 4: Ablation Study (30 seeds)
  Mean Latency WITH V_G:    35.0 триалов
  Mean Latency WITHOUT V_G: 1.0 триалов
  p-value: 4.47e-10
```

## 🔬 Биологические аналоги

- **Акком (2018):** Крысы в Two-Step Task показывают аналогичные MB/MF сигнатуры.
- **Миллер (2017):** Вклад дорсального гиппокампа в MB-планирование.
- **Хас и Редиш (2018):** Пространственная версия Two-Step с VTE-маркерами.

## 📝 Примечания

- **Награды дрейфуют** каждый триал (random walk, σ = 0.01).
- **Changepoint** на триале 1000 (инверсия вероятностей).
- **30 seeds** минимум для публикации.