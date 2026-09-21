# Исполнение joint policy/metric и нового размерного TEST

2026-09-21. Это обновление статусов заметок 54–58, не оценка качества модели.
Старые записи о живых PID описывают прежние срезы, а не текущие процессы.

## Завершённые проверки данных

Collection 53 завершена, её отдельный полный replay/partition audit **passed**:
1494 продолжения, 256 условий. SHA audit:
`15c0be0a866aaf424e2d7d0ddfb2bc11a09185757ab5111063d49d843f785d2a`.
Повторно collection не запускать.

Base supervision 55 завершена: 19189 policy states, но только четыре состояния
с несколькими положительными действиями. Manifest SHA:
`02fea2817015c527928d012677b0280132f3da0923b437bdda56450262dc8f8c`.
Эта скудность альтернатив подтвердила риск имитации единственного witness.

По протоколу 56 добавлены только replay-доказанные равноценные первые ходы.
Новые узлы/сэмплированные траектории этим не создавались; state keys, стоимости
и relative coefficients не менялись. Итог: 8902 multi-positive states,
113555 positive-action slots, максимум 96 положительных действий у состояния.
94362 сертификата: 87953 одинаковых successors, 4759 независимых REDUCE-prefix
перестановок, 1650 adjacent diamonds. Полный отдельный audit **passed**,
session 17851 закончилась с exit 0. Hash manifests/artifacts:

- augmented manifest: `d9526820a69250ba418735b7a66af6c6bc904b42dcc24dfb7cb43126dbbf1f65`;
- certificate audit: `7f898de510953b06c9aac6e7ff98bdcedcbe53dc170e6d4be1deaacb4fa06c76`.

Short-u suffix pool: 6301 views из 161 source witness. После группировки:
exact-0: 5288 records / одна fingerprint-группа; exact-1: 474 / 81;
exact-2: 256 / 15; interval: 283 / 43. Поэтому равномерное sampling по всем
records было бы почти обучением на одном классе; protocol 57 балансирует
класс bounds, fingerprint-группу, затем record.

## Обучение: VERIFIED ACTIVE на проверенном срезе

Unified session **88387**, shell PID 35704, joint worker PID **35705**.
Команды связаны последовательным `&&`: control стартует только после успешного
окончания joint. На срезе joint=112/1024, control ещё не запускался.
Проверены реальные command lines, CPU advance и рост log/checkpoint:
48 → 80 → 96 → 112 steps, CPU 3:53.88 → 7:03.08. Ни один старый worker
не перезапускался. Сам PID не использовать без повторной проверки.

Промежуточный checkpoint 112:
`1434984f185857cd26847c4e82f1e46cef1789835ecdc98be1ac3c391e0d8543`.
SHA совпал с manifest; все 325 model/objective/optimizer tensors конечны.
65 encoder tensors и 14 policy tensors уже изменились; value-head tensors
побитно совпали с исходными. Это подтверждает обучение полной заявленной
части сети, не улучшение расстояний или L10/L1000.

Исправлена до запуска edge-case восстановления: когда атомарный checkpoint
записан, а manifest ещё нет, его step и SHA принимаются совместно; повторное
прерывание сразу после recovery не должно оставлять новый step со старым SHA.
Повреждённые источники, нечисловые веса/optimizer, дубликаты/неверные ресурсы
training states и замаскированные сетью проверенные действия fail-closed.
В новой серии связанных unit tests: **122 passed**. Отдельно три теста нового
prospective-panel audit: **3 passed**. Ruff новых файлов чист.

## Новый TEST: BUILDING, не scored

Session **3499**, worker PID **36256**. Protocol 58, seed=20261024,
4000 attempts, N<=24, глубина конструктивных блоков 24..80. На проверенном
срезе 700 attempts, 43 принятых представления / 43 fingerprint-группы.
Никакая модель ещё не оценивала эту панель. Все 13 старых панелей исключены
из query-групп, включая ранее failed/недостаточные панели.

Подготовлен отдельный `scripts/audit_joint_policy_metric_panel.py`:
replay нового TEST и всех новых source witnesses, независимые determinants,
привязка каждого optimizer state к witness с тем же root/resource context,
пересечение с полной заявленной TRAIN lineage. Исторические audits не
притворяются повторно replayed: они используются как hash-checked evidence;
их 1012 prerequisite hashes уже проверены. Новый итоговый audit ещё не
запущен, поскольку builder не завершён.

После complete panel:

```
.venv/bin/python scripts/audit_joint_policy_metric_panel.py \
  --panel artifacts/mcts-policy-metric-fresh-short-u-test-v1/panel.json \
  --out artifacts/mcts-policy-metric-fresh-short-u-test-v1/independence-audit.json
```

## Следующие действия и границы

1. Следить за реальным продвижением joint, затем control. Не менять frozen
   sources/inputs на ходу; не перезапускать completed arm. Если процесс умер,
   сохранять всё и использовать явный `--resume` после проверки состояния.
2. Дождаться complete TEST и passed independence audit. Не пополнять панель
   после неудачного completion gate и не выбирать checkpoint по ней.
3. Проверить final hashes, finite tensors и одинаковые initial/RNG contracts.
4. Выполнить metric и full-policy cost-MCTS comparison по протоколу 58.
   L10/L1000 улучшение ещё не измерено. Общий Gordian d(K1,K2) требует отдельной
   новой prospective панели; short-u panel сама по себе этого не покрывает.
5. Перенос на любой L/N — глобальная цель, не уже доказанное свойство качества.
   Архитектура 46 не имеет фиксированной таблицы длины/нитей; её размерный
   forward до L16384/N8193 не является решением столь больших узлов.

Старые dirty files и artifacts сохранены. Commit/push, GPU-запуск и promotion
не выполнялись. Состояние Nebius в этой фазе не проверялось: оба новых процесса
локальные CPU; из этого нельзя делать вывод о состоянии прежней GPU VM.

## Следующий срез: оценочный MCTS готов, general TEST строится

2026-09-21, около 03:27 UTC. Joint worker 35705 прошёл 704/1024 шага;
реальный CPU вырос до 21:32.37. Control ещё не стартовал, общий session 88387
жив. Ошибок обучения не зарегистрировано. Это прогресс исполнения, не
обнаруженное преимущество joint над control.

Short-u builder 36256 / session 3499 продолжает фиксированные 4000 attempts;
к 3000 attempts уже набраны 64 query из 64 fingerprint-групп, 32 witnesses с
одним CC и 32 с двумя. Среди выбранных сырых представлений L=69..345/N=3..24:
38 внутри TRAIN envelope, 14 только длиннее L208, четыре только выше N17,
восемь превышают обе границы. Это предварительная структурная статистика
без scoring; final panel и независимый audit ещё ожидаются.

По отдельному prospective протоколу 60 запущен
`build_joint_gordian_panel.py --out artifacts/mcts-policy-metric-fresh-gordian-test-v1`.
Session 57363, PID 39557. Все 2048 attempts завершены: 87 строк / 83 query
fingerprint-группы, exact-0=23, exact-1=28, exact-2=4, interval=32. Gate набора
классов выполнен, но процесс всё ещё формирует/replays size_shift views;
на срезе phase=`writing_panel`, final panel отсутствует. CPU растёт,
запуск не дублировать и завершение не предполагать по одному next_index.

Для общего Gordian TEST использовать тот же независимый lineage audit
`audit_joint_policy_metric_panel.py` с соответствующим panel/out. Он проверяет
общие пары через verify_evaluation_pair, не переименовывает их в u(K).
После complete + passed audit метрики считать существующим
`evaluate_strand_graph_panel.py --bounds determinant` для всех трёх learned
arms на обоих новых panels. До passed audit scoring не запускать.

Реализованы и проверены:

- `scripts/evaluate_joint_cost_mcts.py`: `--arm parent|joint|metric-control|uniform`,
  `--out` и явный `--resume`; pinned протокол 58, только complete/passed panel,
  одинаковые query/resources/seeds, batch-local caches, replay каждого episode
  и решения, отдельный учёт compute, атомарные shards и hash-checked recovery.
- `scripts/audit_joint_cost_mcts.py --run <evaluation-dir> --out <audit.json>`:
  только final evaluation, повторный replay всех continuations и final witness,
  проверка seeds/roots/caps/incumbents/L10/L1000/counts; timings и tree neural
  counters честно обозначены как recorded, не повторно измеренные.
- Дополнительная проверка metric-control: encoder обновляется, параметры
  policy/value heads без gradients остаются прежними, включая AdamW decay.

Последняя новая серия: **32 passed**, Ruff чист. В тестах найдена и исправлена
инициализация счётчика UniformEvaluator до каких-либо scientific MCTS
запусков; ученические данные/веса этим не менялись. Оценочные MCTS ещё
**PREPARED, не LAUNCHED**: ждут frozen short-u panel и passed audit.

## Joint завершён; control и два independence audits активны

2026-09-21. Совместная ветка завершила ровно 1024 шага. Final checkpoint SHA:
`e58270a34569bfad11127ef4f69a8e67bcdcf4825213f88bce03a73a2233ef84`.
Он совпал с progress checkpoint и completed manifest; все input/source pins
сверены с текущими файлами. 325 tensors конечны, 80 optimizer parameter states
имеют step=1024. Изменились 65 encoder и 14 policy tensors; value tensors не
менялись. Optimizer-loop время 1492.42 с, без начального replay источников.
65536 metric samples и 32768 positive-policy samples — не уникальные узлы и
не полное число policy evaluations (relative terms считаются отдельно).

Серийный launcher session 88387 сам перешёл к control, новый PID **42942**,
initial SHA тот же `964199130db5645e41195905f8629636207fcf40e4bcf2a5683b9319fec7044e`.
На проверенном срезе `preparing/metric_source_replay`, CPU 12.74 → 46.12 с;
optimizer ещё не начал шаги. Старый joint PID 35705 завершён, не перезапускать.

Оба новых builder завершились с exit 0, complete=true:

- short-u: 64 validation + 64 lifted views, 64 query-группы,
  SHA `74b158f8734acc2cac05dd9ee07569282da680bbe37f3c68cd70740fb7141d02`;
- general Gordian: 87 validation + 87 lifted views, 83 query-группы,
  SHA `fd8fc390af1db9ceea07531cecc6ca9c9f1d52c50edfd649e9c785511e7669ac`.

Два независимых lineage audits реально запущены одним проверенным script:
general session **77574**, PID **42334**; short-u session **53203**, PID **42818**.
На срезе CPU растёт, final passed artifacts ещё нет; snapshots только живых
процессов, не обещание будущего результата. General audit перепроверяет оба
вида каждого pair; short-u — оба вида каждого explicit-unknot query.

Подготовлен `scripts/audit_joint_training_pair.py`: требует обе final ветки,
точные protocol-57 configs, pinned sources/snapshots/initial/final/progress,
strict model/objective/optimizer loading, finite Adam moments, frozen/updated
группы параметров и одинаковые конечные Python/Torch RNG states. Семь новых
тестов прошли; отдельная более широкая серия **91 passed**, Ruff чист.
Этот paired audit пока **не запускался**, поскольку control не завершён:

```
.venv/bin/python scripts/audit_joint_training_pair.py \
  --out artifacts/mcts-policy-metric-training-audit-v1/report.json
```

Новые TEST scores и MCTS outcomes ещё не получены. Сначала закончить control
и gates, затем выполнить frozen сравнение из 58/60. Улучшение distance
calibration или стоимости развязывания из completed training не следует.

### Исправление несовпадения схемы в independence audit

Первый short-u audit завершился exit 1 до записи результата: ошибочно читал
`optimizer_path_groups` у rollout extension certificate. Это поле существует
в базовом full replay certificate, а extension явно сохраняет его данные как
`rollout_path_groups` (проверено по `verify_rollout_evaluation_extension.py`).
После подтверждения причины остановлен **только** наш read-only general audit
PID 42334, который дошёл бы до той же ошибки; session 77574 exit 143.
Тренировка, builders и какие-либо чужие процессы не прерывались.

Исправленный reader строго проверяет поля/типы исторических certificates
до долгого replay и не подменяет списки счётчиками. Реальный smoke на тех же
входах получил 495 metric, 82 rollout, 809 MCTS-eligible path groups. Добавлен
регрессионный тест различия base/extension schemas; серия **22 passed**.
Старые `.audit.log` сохранены, результаты ими не объявляются passed.

Повторно запущены только audits, с новыми `.audit-retry1.log`:

- short-u: session **78398**, PID **43931**;
- general Gordian: session **92760**, PID **43940**.

На последнем срезе оба процесса живы, CPU вырос ~20 → ~63 с; финальные
passed artifacts ещё не появились. Control PID 42942 перешёл в optimizer и
подтверждённо прошёл 32 → 48/1024 шага с обновлением checkpoint. Ошибка схемы
не затронула data/weights и не ослабила критерий допуска TEST.

## Обе ветки и gates завершены; независимая оценка запущена

2026-09-21. Повторные independence audits завершились exit 0 / **passed**:

- short-u: SHA `5605a0b9fbba86fbe681ea8bfdacecdeae0454e8d64a5fe71b5847664675a8d6`;
- general Gordian: SHA `1637fe98698fab38b97263a8aea16010da3d7efd28fea5f70589df822213a91b`.

Пересечения query с TRAIN lineage и прежними панелями пусты. Проверены все
161 source witness и 19189 positive states / 514 preference terms в 256
условиях. Исторические группы: 495 metric, 82 rollout, 809 MCTS-eligible;
новые source witnesses содержат 178 групп. Fingerprint groups не объявляются
уникальными knot IDs, а references могут быть известны.

Control завершил 1024 шага, source/checkpoint сохранены; серия session 88387
завершилась exit 0. Final control SHA:
`dec2b0d0febfeca1d1b485322438fba16f421f7f906204654a1b7dfc93deddd3`.
Optimizer-loop 859.97 с (начальный replay не включён), 65536 metric samples.
Проверены 271 finite tensors и 62 optimizer states со step=1024; изменился
61 encoder tensor, policy/value parameters остались исходными.

Обязательный training-pair audit **passed**:
`artifacts/mcts-policy-metric-evaluation-v1/training-pair-audit.json`, SHA
`5d7bac753a9b6b0de10a7d2f6111f5d12289dfa54717b129ddede01dc13caf75`.
Подтверждены одинаковые initial tensors, final Python/Torch RNG streams,
fixed configs/sources/data, snapshots, final/progress hashes и finite weights.
Это проверка сопоставимости исполнения, не результат TEST.

`run_joint_policy_metric_evaluation.py` реально запущен: session **64894**,
controller PID **46131**. Он сначала проверял наличие живого control worker,
затем автоматически перешёл к scoring после завершения training и обоих
data audits. Не запускает/не останавливает обучение, не доверяет старым PID,
не создаёт замену пропавшему частичному evaluator без разбора состояния.
12 тестов controller/gates/metric-output checks прошли.

Фиксированные 15 задач: paired-training audit; шесть metric оценок
(parent/joint/control × short-u/general, determinant bounds, оба вида в
каждом отчёте); четыре full-policy MCTS запуска (три модели и uniform),
после каждого — отдельный replay audit. Sources/checkpoints/panels pinned;
не менять их во время active sequence.

На проверенном срезе завершена задача 1/15. Активен **metric-parent-short-u**,
PID **47668**, CPU 27.05 с, свежий собственный log. Финального metric JSON
ещё нет. Не считать queued задачи выполненными и не выбирать модель по
появившемуся первым частичному результату.

Подготовлен `summarize_joint_policy_metric_evaluation.py`: принимает только
все 15 завершённых hash-verified задач с нужными audits, проверяет совпадение
roots/resources/models, сохраняет constant controls с тем же равномерным
взвешиванием групп, pairwise errors, censored search outcomes, отдельные λ и
raw-size strata. Common-only cost не скрывает newly/lost solutions. Четыре
новых summary-теста и 12 controller-тестов: **16 passed**, Ruff чист.
Этот final consumer пока **не запускался**:

```
.venv/bin/python scripts/summarize_joint_policy_metric_evaluation.py \
  --manifest artifacts/mcts-policy-metric-evaluation-v1/manifest.json \
  --out artifacts/mcts-policy-metric-summary-v1
```

Ни lower train loss, ни пройденный paired audit не доказывают ещё меньших
TEST errors/L10/L1000. Требуемые независимые результаты сейчас вычисляются.

## Следующий проверенный срез: 3/15, исходная модель оценена

2026-09-21. Завершены paired audit и обе metric оценки parent. Controller
PID 46131 жив; новый worker **metric-joint-short-u**, PID 49566, наблюдался
с ростом CPU 27.57 → 68.65 с. Старый PID 47979 не перезапускался: его
general-Gordian job действительно завершён и появился итоговый JSON.

Итоговые parent outputs:

- short-u SHA `757c7f1a13c78519b653196638f8af040490958c7c562aab24411313556ed6bf`;
- Gordian SHA `90db0338381a29f6330b4c4f7e611aaa52a4fc2bfaf25f312953a78d21b052fe`.

Это ещё не сравнение трёх моделей. Final consumer не запускался, источники
active evaluation не изменялись. Добавлен пятый summary-тест: целая synthetic
матрица отчёта, отдельные λ, цензурированные пропуски и пустые размерные
страты. Все пять summary-тестов прошли; Ruff чист.

### Уточнение области действия уже замороженного поиска

По протоколу 58 CC cap равен числу crossing changes в reference witness.
Это одинаковое ограничение для всех моделей, но **не доказательство
минимальности L10/L1000**: особенно для L10 путь с дополнительным CC и
существенно меньшим числом остальных ходов может быть дешевле, но исключён
из текущего поиска. Не менять этот cap посреди серии. Final summary явно
сохраняет данное ограничение. Отдельный будущий эксперимент с расширенными
caps должен сравнивать модели заново на одинаковых ресурсах и не подменять
собой текущий результат. Незавершённые попытки не становятся негативными
метками ни при одном из этих ограничений.

Следующий read-only verifier уже подтвердил **4/15** завершённых outputs:
joint short-u тоже готов, controller перешёл к `metric-joint-gordian`,
worker PID 49903. Все frozen input hashes неизменны, завершённые outputs
повторно проверены через `verify_job`. Расширенный набор summary/controller/
MCTS/training-pair tests: **35 passed**. Сравнительный final report пока не
вычислен; ещё впереди control metric и четыре поисковых arms с replay.

## Найдена ошибка reader перед поиском; восстановление подготовлено

Дополнительный read-only `audit_cost_resource_envelope.py` использовал тот же
`build_roots`, что запланированный поиск, и завершился до создания output:
`ValueError: search query is not an explicit-unknot task`. Реальная запись
unknot имеет `word=[]`, `strands=1`, **`cyclic_band_generators=false`**;
старый reader сравнивал весь словарь только с двухполевой формой. Метрика
использует другой reader и ошибкой не затронута.

Регрессионный тест с обоими форматами подтвердил причину: минимальная форма
проходит, сериализованная полная форма падает. Предусмотрено только принятие
двух равнозначных ordinary-unknot форм, без изменения панели/семантики/caps.
Во время живого frozen controller исходный evaluator ещё **не менялся**.

Подготовлен `recover_joint_evaluation_schema.py`. После естественного
завершения всех семи pre-search jobs и остановки первого поиска на этой
конкретной ошибке он допускает лишь побайтово заданный source patch; проверяет
отсутствие живых workers и поисковых артефактов, копирует семь неизменных
итогов в новый каталог, сохраняет hash-linked failed lineage и старый source
snapshot. Затем прежний controller можно продолжить с `--resume`. Это не
перезапуск обучения и не повторный scoring по выбранным результатам.
Семь recovery tests, включая полное копирование toy lineage и отказ после
повреждения копии, прошли. Итоговый consumer расширен проверкой reuse pins.

### Частичный законченный срез метрики short-u (не promotion)

Все три validation short-u outputs уже готовы: 64 query-группы, одинаковая
оценка по группам. Parent error=0.3417663, joint=0.2919444,
metric-control=0.2953815. Константа 1 даёт 0.125, так что снижение средней
ошибки не означает превосходство этого простого baseline. Точные u=1: 32
строки, MAE 0.5637707 / 0.4663290 / 0.4685020; точные u=2: восемь строк,
MAE **0.1011371 / 0.1317791 / 0.1364086** — ухудшение у обеих новых веток.
General metric и поиск ещё не завершены; checkpoint/условия по этому срезу
не выбирались и не изменялись.

## Schema recovery выполнен; настоящий поиск продолжен

Все семь pre-search outputs завершены. Старый controller session 64894
завершился exit 1; PID 46131 и первый search worker 53589 отсутствуют.
Последний остановился строго на указанной проверке endpoint, до создания
`mcts/` и каких-либо поисковых результатов. Старый manifest и failure log
сохранены без изменения:

- manifest SHA `9e0947249be90d3d0fc812764e9a711fb7667fd4bf6e899627016445aef6ca39`;
- failure log SHA `c33f561eabc228d6a805d71061e58d3794e96c3eabe02153ed7080165521495e`.

Разрешённый patch evaluator: прежний SHA
`9c5dabc87b2d21f11f7ea538d4034af501134f3173b2cff91769eb912cb7b7be`,
новый `d42009f9fb55928ef60a6ba42ce3b9ab9f259f21bb33cea1ba63e4901bc98510`.
Вся разница — дополнительная равнозначная форма explicit ordinary unknot.
Четыре теста продолжают отвергать другой endpoint, band=true и неизвестное
поле. Совместно с recovery/summary/envelope/controller: **44 passed**.

Recovery завершился exit 0 и сохранил семь byte-identical outputs в
`artifacts/mcts-policy-metric-evaluation-schema-v2`. После этого запущен:

```
.venv/bin/python -u scripts/run_joint_policy_metric_evaluation.py \
  --out artifacts/mcts-policy-metric-evaluation-schema-v2 --resume
```

Session **71142**, controller PID **53921**, search `mcts-parent` PID **53951**.
Новый manifest/roots/source snapshot уже созданы; на первом проверенном срезе
CPU worker=95.56 с, `next_condition=0`: первый batch ещё не завершён. Это
живой поиск, не claim о готовых solutions. Не менять его frozen sources.

Final consumer теперь нужно направлять на **schema-v2/manifest.json**.
Он проверяет и сохраняет hashes исходной failed lineage, точную source
поправку, совпадение всех семи повторно использованных результатов. Сам
final consumer ещё не запущен.

### Уточнение CC cap по завершённой арифметической проверке

`artifacts/mcts-policy-metric-resource-envelope-v1/report.json` завершён;
SHA `43d6808969b6e18d8a0148e1e556186d81583f673698f7e0ab8245a36176fc9b`.
Для обычных semantic actions каждый non-CC ход удаляет максимум две буквы
и максимум одну нить; CC входит в moves и ничего из этого не удаляет.
Поэтому при начальных L,N и c crossing changes:

```
q = max(ceil(L/2), N-1)
m >= c + q
L_lambda >= (lambda+1)*c + q
```

Для строгого улучшения integer reference cost B необходимо
`c <= floor((B-1-q)/(lambda+1))`. В обоих λ=10/1000 **0 из 64** корней
допускают арифметическую возможность улучшить B с c сверх текущего cap.
Следовательно, общая осторожность о дополнительных CC не является
обнаруженным дефектом этой панели: здесь CC cap безопасен именно для
улучшения reference cost. Move/growth caps, horizon и ограниченный набор
кандидатов всё ещё могут исключать улучшения. Это не доказательство
оптимальности найденного поиска и не отрицательные метки для его неуспехов.
Final consumer сверяет эти строки со своими roots и включает уточнение.

### General Gordian: законченный metric срез

83 query-группы / 87 пар; error parent/joint/control:
0.5306414 / 0.4758786 / 0.5046595. Константа 1: 0.2891566.
С determinant bounds точных d=1 стало 48 (у исходной signature-only панели
их было 28); d=0: 23, d=2: четыре. MAE d=2 остаётся большой:
1.6168164 / 1.5667469 / 1.5500398. Таким образом, joint улучшил обе средние
ошибки относительно parent, но не превзошёл константу 1 и не исправил
калибровку общих коротких расстояний. О пользе новой policy для поиска
вывод до завершения всех четырёх arms/replay audits не сделан.

Первый durable search prefix (условия 0..7) отдельно прочитан через
`committed_rows`, hashes/shards и все frozen inputs проверены; recovery
receipt тоже повторно проверен. Затем manifest продвинулся до **16/128**:
два shards, 96 trajectories, 11308 continuation moves, 2012 policy batches,
51 tree embedding evaluation, пока ноль решений. Committed batch time
164.02 с. CPU worker вырос 95.56 → 205.24 с. Это промежуточный parent arm,
не итоговый отрицательный результат о модели/embeddings. Все 44 связанные
теста после последних изменений прошли, Ruff чист.

Следующий проход продолжил live search без изменения исходников: parent
прошёл как минимум 48/128 условий, CPU worker=556.70 с, шесть сохранённых
пакетов; 288 continuations / 34308 policy neural evaluations. Пока ноль
решений, это промежуточный срез. Отдельно подготовлено побитово эквивалентное
ускорение graph preprocessing (заметка 61); в текущие frozen arms оно не
внедрялось. Source/input pins и hashes committed search prefix перепроверены.

## Parent search завершён, replay audit запущен

128/128 условий (64 корня × две λ), **0 решений**. Финальный manifest SHA
`4c924100619a501826b2239d08a8d4439ab12518bb77bf283a728fcad07989cd`.
768 продолжений, 79328 continuation moves/policy neural evaluations,
79968 полных ходов с tree prefix, 13474 policy batches, 385 tree embedding
evaluations. Время committed batches — 1003.69 с. Это не время всей серии
с prep/audit, и parent timing включал указанную в заметке 61 диагностику.

Старый search PID 53951 завершён. Controller 53921 сам перешёл к
`mcts-audit-parent`, worker PID **58308**, CPU проверенно рос до 135.69 с.
На этом срезе audit ещё не выдал итоговый JSON; серия **8/15**, не 9/15.

Все 384 продолжения каждого λ остановились по `move_cap`. Средняя длина
начала 191.96, конца 179.11 (λ10) / 158.25 (λ1000); среднее продолжение
103.29 хода. Policy λ10 выбрала 2536 REDUCE, 15805 COMMUTE, 9264 BRAID,
5799 STABILIZE, 5751 DESTABILIZE; λ1000 — 6469 REDUCE, 26126 COMMUTE,
6376 BRAID, 41 STABILIZE, 146 DESTABILIZE. Частоты не являются доказательством
плохости конкретных ходов; неуспехи не преобразованы в негативные метки.

Final summary теперь сохраняет для **каждого** arm и λ одинаковую
диагностику: причины остановки, виды действий, средние start/end lengths
и число continuation moves, отдельно от tree prefixes. Это должно помочь
интерпретировать сравнение, а не подбирать новый checkpoint по TEST. Все
49 связанных tests прошли, Ruff чист. Новая joint policy и control ещё
не оценены поиском; вывод об эффекте их обучения преждевременен.

Replay audit parent завершён **passed**, SHA
`4a508d7da6ea6cbdd328860624b5c660da0fd674f35a51b4c3c617fb9c87328f`:
перепроверены все 128 условий / 768 episodes / 79328 continuation moves /
79968 full-path moves / 13423241 action scores. Нулевой результат parent
не связан с отсутствием replay; это подтверждённый результат данного
ограниченного протокола, не вывод о неразрешимости состояний.

Controller автоматически начал **mcts-joint**, PID **59268**, parent 53921.
Manifest/roots/source snapshot созданы, CPU вырос 19.03 → 93.08 с. Первый
batch ещё не завершён (`next_condition=0`). Независимый read-only verifier
подтвердил все **9/15** законченных jobs, recovery receipt и неизменность
frozen inputs. Следом остаются joint audit, metric-control + audit,
uniform + audit; final comparative summary по-прежнему не запускался.

Joint продвинулся до **80/128**, CPU PID 59268=774.85 с, пока ноль решений.
Текущий committed prefix и frozen input pins перепроверены. Одновременно
завершён TRAIN-only policy backward/device probe (заметка 62), без шагов
optimizer и без изменения этих весов/поиска. MPS численно совместим, но
не bit-reproducible; на batch 128 преимущество MPS bulk над CPU bulk только
около 1.09×. Исходный failed exactness probe сохранён, повторная проверка
отделила вариативность самого MPS от integer adjacency. Все три отчёта
связаны hashes, source/data pins повторно проверены. Это не полный training
benchmark и не свидетельство решения задачи embeddings.

## Joint search и его replay audit завершены

128 условий, **2 solved conditions**, но это **один query 53 из 64** в обоих
режимах стоимости (λ — вес CC, не число симуляций). Исходная модель не решила
его ни в одном режиме. Query L=70, N=3, внутри TRAIN size envelope; это не
свидетельство экстраполяции на большие L/N.

Оба новых witness отдельно воспроизведены. При λ10: c=1, m=39, L10=49,
L1000=1039; при λ1000: c=1, m=45, L10=55, L1000=1045. Известный reference:
c=1, m=37, L10=47, L1000=1037. Покрытие улучшилось относительно parent,
но найденные пути не улучшили reference cost. Common-success delta с parent
не определена: у parent нет успешных query.

Joint final manifest SHA
`d3f9c774d30d90f27ed2a824f1736f8a78f838c12e1b28986518ec69c21ccb5b`;
replay audit **passed**, SHA
`6de26b8d3f3eb81245d55f780e3d068bd32f78b47026913971fcd44d3fcbff68`.
Audit подтвердил 128 conditions / 766 continuations / 79156 continuation
moves / 79796 full-path moves / 11179264 action scores / 2 solutions.
Committed-batch time 919.38 с, не изолированный performance benchmark.

Серия перешла к **11/15**, `mcts-metric-control`, PID **64169** с parent
53921, свежий manifest/roots/snapshot и CPU=91.79 с. На первом срезе
`next_condition=0`, первый batch ещё не завершён. Control и uniform нужны,
чтобы не приписать одному positive-policy loss ещё не изолированный эффект.

Уточнение интерпретации размерных страт внесено в final consumer: within
(38 query) имеет move slack 0..27, mean cap/reference=1.0848; length-only
(14), strands-only (4), both (8) имеют нулевой slack. Это не одинаковая
относительная сложность бюджетов по размерам. Заметка 63 выводит точную
обязательную часть semantic cost и фиксирует следующий read-only audit;
сама по себе предварительная проверка по roots не названа готовым audit.

## Завершённый semantic-cost audit, 2026-09-21

`artifacts/mcts-policy-metric-cost-floor-audit-v1/report.json` завершён со
статусом **passed**, SHA256
`7f93115a4ffd2daf389a68915a475f37d7dbaa75cf90fe87dda94b0fed75d1f0`.
После завершения повторно проверены все привязанные входы и исходники.
Перепроиграны 64 reference witnesses, их префиксы и обе joint solutions;
signature bounds пересчитаны, determinants проверены независимым вычислением.

У всех 64 reference нулевой excess в тождестве из заметки 63. У 40 совпали
доказанная нижняя граница и стоимость reference при обоих весах CC: это
оптимумы **данной системы semantic moves для данных начальных представлений**.
У остальных 24 такой сертификат отсутствует. Их необходимый move envelope
для стоимости не выше reference превышает текущий move cap; это не
доказательство существования пропущенного более дешёвого пути.

Для query 53 joint имеет доказанный разрыв до оптимальной стоимости +2
при λ=10 и +8 при λ=1000. Это два условия на одном query, не два независимых
успеха и не улучшение известного reference. TEST-сертификаты не использованы
в обучении и не меняют текущие search budgets или метки неуспехов.

В unit fixture удалён недопустимый в том состоянии PASS; production replay
и правила допустимости не менялись. 30 связанных тестов cost identity,
strand-graph architecture и lazy actions прошли. На последнем срезе control
PID 64169 продвинулся с 96 до 104/128 conditions, CPU time с 15:34.82 до
16:03.35; найденных решений пока 0. Серия ещё не завершена.

## Metric-control завершён; uniform активен

Control завершил 128/128 conditions, 767 continuations, 79255 continuation
moves, 79895 full-path moves, 13517353 candidate action scores. Final manifest
SHA `f025b5dc17ce27ea6d0d5cb20de1d99c0df3b3e36f8d2f9362b36016093a16a5`;
отдельный replay audit **passed**, SHA
`1b6ec42ce6b8f0cc9f7ef4917b303464718d139433dfd1b5477bd14c71eb796b`.
Все pinned inputs трёх завершённых learned arms перепроверены после аудита.

Единственный успех control — query 53 / condition 107 / λ=1000,
L1000=1055. У joint на том же условии L1000=1045, то есть на 10 меньше;
разрыв до доказанного оптимума соответственно +18 и +8. При λ=10 joint
нашёл решение этой query, а control — нет. Это одно общее успешное условие
и один новый успех, не массовое улучшение качества или статистическая
значимость. Parent не решил ни одного условия.

Final consumer расширен сертификатами cost floor: проверяет привязку audit
к panel/manifest, coverage, fingerprints, q0, повторно выводит bounds и
отличает gap до нижней границы от доказанного regret. Неуспехи сохраняет
как пропуски. Расширение проверено на трёх настоящих completed arms;
84 связанных теста прошли, Ruff чист. Итоговый отчёт не создаётся до
завершения всех 15 задач, включая uniform и его replay audit.

Controller PID53921 завершил 13/15 задач и запустил uniform PID69442.
На проверенном срезе worker имел CPU time 2:19.47, elapsed 2:21, 99.4% CPU;
durable log продвинулся до 16/128 условий. Замороженные исходники, параметры
поиска и обучения не менялись; повторного worker нет.

Read-only исследование следующего источника разнообразия — заметка 64.
Две реальные SQLite sidecars проверены по схеме, metadata, count queries и
SHA. Выявлены несовместимые mirror-quotient CC semantics и state-level
policy split; данные не импортированы и не названы leakage-safe TRAIN.

## Вся серия завершена, итоговый consumer проверен

Все15/15 jobs завершены; uniform дал0/128 solved conditions, replay audit
passed. Controller и его workers больше не живы по проверке процессов;
новые поисковые запуски не создавались. Final controller SHA
`b349db17906ac74f77b7ac593f5724aee58b30c7c38116a17c1fdcdd2840e180`.

Выполнен final-only consumer:
`artifacts/mcts-policy-metric-summary-v1/report.json`, SHA
`f336ee84f8baa41c4af7a7ea85240a9a5a5004001fb0a72c6a4f2ebea5120c06`;
все его inputs и sources повторно проверены после завершения. Научные
результаты, отрицательные стороны и следующий этап — заметка66. Это
завершение текущего эксперимента, не достижение общей цели arbitrary-size RL.

Дополнительно завершены fixed512-path ordinary-graph probe и отдельный
source/partition/replay audit:283 допустимых TRAIN-owner traces,34 invariant
groups,13 exact2 pairs, без совпадений с query-группами15 historical panels.
См. итоговое дополнение заметки64; исходный протокол65 не переписан.
Эти записи — пары общих Gordian-расстояний, не full unknotting witnesses.
