# README_Stage_3.1A

## Статус

Этап **Stage 3.1A** закрыт как **calibrated baseline** для пространственного выбора при равной награде и различии по observability / exposure.

Финальная рабочая конфигурация для закрытия этапа:
- `exposure_q_bias = 0.35`
- `eps = 0.05` в timeout-fallback

Контрольная конфигурация для сравнения:
- `exposure_q_bias = 0.40`

Основания: калибровочный sweep по `exposure_q_bias` и полный baseline-анализ для `0.25 / 0.30 / 0.35 / 0.40`. (Table_3_1C_BiasSweep.csv), (Stage3_1A_Baseline_Report 0.25.md), (Stage3_1A_Baseline_Report 0.30.md), (Stage3_1A_Baseline_Report 0.35.md), (Stage3_1A_Baseline_Report 0.40.md)

---

## Цель этапа

Stage 3.1A решал не задачу обучения карте лабиринта, а задачу **junction choice under exposure asymmetry**: агент должен был принимать стохастическое решение на развилке между `open` и `covered` при равной reward-структуре, но различной экспозиции среды.

Практическая цель этапа состояла в том, чтобы:
1. восстановить принципиальную стохастичность выбора;
2. отделить deliberation от принудительного timeout-commit;
3. получить устойчивый, но не вырожденный bias в сторону `covered`;
4. зафиксировать калиброванный baseline перед переходом к следующему содержательному этапу.

---

## Исходные проблемы, устранённые на этапе

В процессе отладки были устранены следующие архитектурные дефекты.

### 1. Преждевременный commit на junction

Выбор пути происходил на том же тике, на котором агент только входил в junction. Это схлопывало фазу deliberation и делало поведение фактически детерминированным.

Решение:
- введён барьер входа на junction;
- commit запрещён на тике прибытия;
- движение после commit отделено от самого акта commit.

### 2. Стационарная deliberation без накопления evidence

Ранее агент повторно семплировал почти одну и ту же policy до timeout. Это давало не deliberation, а `repeated sampling until timeout`.

Решение:
- введён **signed evidence accumulator**;
- commit переведён на `bound crossing + urgency collapse`;
- `commit_latency` и `junction_exit_tick` стали считаться по реальному моменту решения и выхода из junction.

### 3. Искажённое summary logging

До правок `mode_at_junction` и часть junction-метрик брались слишком поздно, уже после выхода из критической фазы.

Решение:
- `mode_at_junction` защёлкивается на junction;
- `commit_reason` логируется явно (`bound` / `timeout`);
- trial summary перестроен на корректные временные точки.

### 4. Асимметричный timeout fallback

На промежуточной стадии timeout-fallback систематически уводил выбор сначала в `open`, затем после локальной правки — в `covered`. Оба варианта были интерпретационно неверны.

Решение:
- введён порог `eps = 0.05`;
- при малом остаточном evidence timeout перестал быть скрытым deterministic default;
- fallback снова стал интерпретируемым как слабое завершение deliberation, а не как жёсткая подмена policy.

---

## Что именно было проверено

### Bias sweep

Был проведён sweep по `exposure_q_bias`.

Полный sweep: `0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6`. (Table_3_1C_BiasSweep.csv)

Уточняющий sweep после введения `eps = 0.05`: `0.20, 0.25, 0.30, 0.35, 0.40`. (Table_3_1C_BiasSweep.csv)

Ключевой результат: между `0.25` и `0.30` обнаружен **переход режима**. До `0.25` система остаётся более медленной и timeout-heavy; начиная с `0.30` covered-bias становится устойчивым, а deliberation-показатели выходят на почти плоское плато. (Stage3_1A_Baseline_Report 0.25.md), (Stage3_1A_Baseline_Report 0.30.md), (Stage3_1A_Baseline_Report 0.35.md), (Stage3_1A_Baseline_Report 0.40.md)

### Seed-level stability

Эффект проверялся на `50 seeds × 100 trials = 5000 trials` для baseline-run. Разброс по seeds остаётся умеренным, без зависимости результата от нескольких выбросов. (Stage3_1A_Baseline_Report 0.35.md)

### Commit reason analysis

Отдельно анализировались:
- общая частота `bound` vs `timeout`;
- распределение `path_choice` внутри `bound` и `timeout`;
- различия в `commit_latency`, `junction_pause_duration`, `reorientation_count` по типу commit и по выбранному пути.

### Block dynamics

Все триалы разбивались на блоки `1–25`, `26–50`, `51–75`, `76–100`, чтобы проверить наличие learning drift внутри сессии.

Результат: выраженного межблочного дрейфа не обнаружено. Это подтверждает, что Stage 3.1A является baseline без learning-of-maze dynamics. (Stage3_1A_Baseline_Report 0.35.md)

---

## Финальный baseline: почему выбран именно 0.35

### 0.25

Параметр `0.25` уже даёт covered-bias, но система остаётся слишком медленной:
- `P(covered) = 0.6190`
- `mean_commit_latency = 4.4764`
- `P(timeout) = 0.1820`
- `mean_reorientation_count = 1.7342`

Это полезный режим для сравнения, но как baseline он слишком вязкий. (Stage3_1A_Baseline_Report 0.25.md)

### 0.30

Параметр `0.30` даёт более быстрый и уже устойчивый режим:
- `P(covered) = 0.6362`
- `mean_commit_latency = 3.4252`
- `P(timeout) = 0.1206`
- `mean_reorientation_count = 1.1964`

Это минимальная точка после перехода режима. (Stage3_1A_Baseline_Report 0.30.md)

### 0.35

Параметр `0.35` после введения `eps = 0.05` даёт наиболее удачный компромисс:
- `P(covered) = 0.6596`
- `mean_commit_latency = 3.4232`
- `P(timeout) = 0.1210`
- `mean_reorientation_count = 1.1952`
- `covered_rate_std_across_seeds = 0.0504`

При этом `covered` и `open` присутствуют и в `bound`, и в `timeout`, то есть fallback снова стал симметричным и интерпретируемым. (Stage3_1A_Baseline_Report 0.35.md)

### 0.40

Параметр `0.40` усиливает covered-bias:
- `P(covered) = 0.6806`
- `mean_commit_latency = 3.4136`
- `P(timeout) = 0.1174`
- `mean_reorientation_count = 1.1832`

Это уже рабочий режим, но как baseline он сильнее смещён в сторону `covered` и поэтому оставлен как контрольная конфигурация, а не как основной стандарт этапа. (Stage3_1A_Baseline_Report 0.40.md)

### Вывод

`0.35` выбран как **середина пост-переходного плато**:
- bias уже устойчив;
- deliberation ещё не схлопнута;
- timeout не доминирует;
- fallback симметричен;
- разброс по seeds остаётся умеренным.

---

## Что Stage 3.1A теперь показывает надёжно

### 1. Поведение больше не детерминировано

На этапе отладки была устранена исходная проблема полного коллапса в один и тот же path. После исправлений выбор остаётся стохастическим и варьирует между `open` и `covered` при всех рабочих значениях bias.

### 2. Возник устойчивый covered-bias

Для baseline `0.35 + eps 0.05` агент выбирает `covered` в `65.96%` триалов. Это уже не шум и не artefact одного seed, а устойчивый эффект на серии `50 × 100`. (Stage3_1A_Baseline_Report 0.35.md)

### 3. Deliberation жива, но не вырождена

Для baseline:
- `mean_commit_latency = 3.4232`
- `mean_junction_pause_duration = 4.4232`
- `mean_reorientation_count = 1.1952`

То есть решение не схлопнуто в мгновенный argmax и не сводится к почти полному timeout-ждению. (Stage3_1A_Baseline_Report 0.35.md)

### 4. Timeout больше не является скрытым default

После введения `eps = 0.05` timeout не редуцируется ни к `open-default`, ни к `covered-default`. Это важно для интерпретации данных и для перехода к следующему этапу. (Stage3_1A_Baseline_Report 0.35.md), (Stage3_1A_Baseline_Report_no_eps 0.35.md)

### 5. Внутрисессионного learning drift не обнаружено

Block dynamics для `0.35` почти плоские:
- `P(covered)` держится в диапазоне `0.692–0.722`
- latency и reorientation тоже меняются слабо

Это подтверждает, что Stage 3.1A — baseline для **exposure-biased junction choice**, а не стадия обучения карте лабиринта. (Stage3_1A_Baseline_Report 0.35.md)

---

## Ограничения этапа

### 1. `mode_at_junction = explore` фактически константен

В baseline-run `P(explore at junction) = 1.0000`. Это означает, что на Stage 3.1A ещё не продемонстрирована полноценная вариативность Gate именно на junction. В текущей версии вариативность в основном локализована в policy layer и в junction commit wrapper. (Stage3_1A_Baseline_Report 0.35.md)

### 2. Stage 3.1A не тестирует learning of reward map

Этап не показывает обучение “какой путь выгоднее”. Он показывает bias в выборе при равной reward-структуре и различной экспозиции среды.

### 3. Биологическая интерпретация пока не является целью

Показатели `reorientation`, `commit_latency`, `pause_duration` уже можно использовать как VTE-proxy на инженерном уровне, но делать сильную биологическую интерпретацию по Stage 3.1A пока рано.

---

## Что считается завершённым

Этап считается закрытым, потому что выполнены следующие условия:

1. исправлена state-machine logic на junction;
2. введена рабочая deliberation architecture (`signed evidence + bound crossing + urgency collapse`);
3. устранён deterministic collapse поведения;
4. проведён bias sweep и найден калиброванный baseline;
5. baseline проверен на серии `50 seeds × 100 trials`;
6. выполнен полный baseline-анализ по seed-level spread, commit reason, path choice и block dynamics.

---

## Что переходит в следующий этап

В следующий этап переходят:

1. **Stage 3.1B: threat / reward conflict**
   - переход от чистого exposure-bias к конфликту между reward и threat;
   - проверка, сохраняется ли структура deliberation при реальном конфликте ценностей.

2. **Аналитика абляций**
   - `NoX`
   - `NoVG`
   - `NoVp`
   - one-shot / temporal-state manipulations

3. **Уточнение роли Gate**
   - отдельная проверка, даёт ли Gate вариативность именно на junction, а не только на уровне post-gate policy.

---

## Финальная формулировка статуса

**Stage 3.1A завершён успешно как calibrated baseline для exposure-biased stochastic junction choice.**

Финальная baseline-конфигурация этапа:
- `exposure_q_bias = 0.35`
- `eps = 0.05`

Контрольная конфигурация:
- `exposure_q_bias = 0.40`

Этого достаточно для перехода к следующему содержательному этапу.
