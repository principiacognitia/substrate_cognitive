# Stage 2: Experimental Validation Package

**Статус:** 🟡 В разработке (Ожидаемое завершение: 4 недели)

## 🎯 Цели Stage 2

Stage 2 представляет собой полный пакет экспериментальной валидации архитектуры Gate-Rheology на двух канонических задачах:

1.  **Two-Step Task (Daw et al., 2011)** — Доказательство различимости MB/MF стратегий и существования гистерезиса переключения режимов.
2.  **Block-Reversal Task (Le et al., 2023)** — Доказательство роли V_G (инерция контроля) и V_p (инерция действия) в когнитивной ригидности и персеверации.

## 📋 Критерии готовности (Stage 2 Ready)

- [x] Среда Two-Step реализована и валидирована (Level 1)
- [x] MB/MF сигнатуры различимы (Level 2, p < 0.001)
- [x] V_G гистерезис продемонстрирован (Level 3, Latency > 5 триалов)
- [x] Абляция V_G показывает значимый эффект (Level 4, p < 0.001)
- [ ] Reversal Task реализована и валидирована
- [ ] Двойная диссоциация V_G vs V_p показана
- [ ] CV Log-Likelihood вычислен для всех моделей
- [ ] Все графики для статьи сгенерированы

## 🗂️ Структура

```
stage2/
├── core/                   # Общие модули
│   ├── gate.py             # Мульти-гейт арбитраж
│   ├── rheology.py         # Обновление V_G, V_p
│   ├── baselines.py        # MF/MB/Hybrid/NoVG агенты
│   └── logger.py           # Единый формат логирования
├── twostep/                # Two-Step Task
│   ├── env_twostep.py
│   ├── agent_twostep.py
│   ├── config_twostep.py
│   └── debug_*.py          # Скрипты отладки (Levels 0-4)
├── reversal/               # Reversal Task
│   ├── env_reversal.py
│   ├── config_reversal.py
│   └── run_reversal.py
├── analysis/               # Анализ и визуализация
│   ├── figures.py          # Генерация графиков для статьи
│   ├── cv_likelihood.py    # Cross-validation log-likelihood
│   └── bio_comparison.py   # Сравнение с биологическими данными
└── README.md               # Этот файл
```

## 🚀 Запуск экспериментов

```bash
# Two-Step Task
python -m stage2.twostep.debug_sanity_check    # Level 2: MB/MF сигнатуры
python -m stage2.twostep.debug_integration     # Level 3: V_G динамика
python -m stage2.twostep.debug_ablation        # Level 4: Абляции

# Reversal Task
python -m stage2.reversal.run_reversal         # Прогон агента
python -m stage2.reversal.debug_ablation       # Абляции в Reversal

# Анализ
python -m stage2.analysis.generate_all_figures # Все графики для статьи
```

## 📊 Ожидаемые результаты

| Эксперимент | Метрика | Ожидаемое значение |
| :--- | :--- | :--- |
| **Two-Step MB** | interaction coef | > 0.3, p < 0.01 |
| **Two-Step MF** | interaction coef | ~ 0.0, p > 0.10 |
| **V_G Ablation** | Latency (Full vs NoVG) | 30× разница, p < 0.001 |
| **Reversal** | Perseverative Errors | Full > NoVp > NoVG |
| **Reversal** | Trials to Criterion | Full > NoVG |

## 📄 Сопутствующая документация

- [Two-Step Task Spec](./twostep/README.md)
- [Reversal Task Spec](./reversal/README.md)
- [Mapping Document](../docs/mapping.md)

## ⚠️ Важные замечания

1.  **Параметры агента фиксированы** для всех задач. Перенастройка под задачу запрещена (тест универсальности).
2.  **Все эксперименты** должны запускаться с минимум 30 seeds для статистической мощности.
3.  **Логи сохраняются** в `logs/` в едином формате (CSV + meta.json).