# Исполнение сбора TRAIN: протокол 53

2026-09-21. Статус на момент этой записи: **VERIFIED ACTIVE**, не completed.
Обучение новых весов пока не запускалось; выполняется сбор и проверка
траекторий frozen actor для следующего joint-этапа.

## Запуск

`scripts/collect_cost_mcts_training.py --out artifacts/diverse-cost-mcts-training-v1`

Unified exec session 22992, PID 22236. PID — только ориентир для последующего
наблюдения, не доказательство, что работа всё ещё жива. Проверены command
line, рост process CPU с 9.03 с до 4:13.81, свежий manifest и семь завершённых
shards. На последнем проверенном срезе выполнено 56 из 256 root/weight задач,
326 продолжений, 23957 ходов продолжений, 7 найденных решений по условиям.
185 траекторий допущены по path ownership, в том числе 10 успешных;
пять условий имеют все записанные траектории внутри TRAIN.
Эти счётчики промежуточные и не являются финальной оценкой качества.

Выбраны заранее 128 разных fingerprint-групп, по одному представлению каждой,
парно λ=10/1000. Фактический диапазон корней: L=13..208, N=3..17.
Источник уже содержит конструктивные удлинения; отсутствие новых lifts в
этом сборе не делает все длинные корни принципиально более сложными узлами.
Веса и исходники frozen, snapshot находится в output/source_snapshot.

## Что готово для следующей проверки

`scripts/audit_cost_mcts_training.py` — отдельный gate после status=completed.
Он требует полное покрытие, перепроверяет SHA, roots и reference witnesses,
каждый tree-prefix и rollout, точные CC endpoint fingerprints, eligibility,
ресурсы и найденные witnesses. Отдельно определяет допустимость лучшего
witness: флаг `all_episodes_train_owned` сам по себе не доказывает допуск
свидетельства, найденного непосредственно деревом вне записанных rollout.
Подсчитывает условия с разными стоимостями успешных допустимых путей.

До запуска сбора прошли 64 связанных теста. Позже отдельная серия новых
collection/audit/rollout тестов: 34 passed; Ruff новых файлов чист.
Фактический полный audit ещё НЕ запущен: collection остаётся активным.

После завершения, без перезапуска collection:

```
.venv/bin/python scripts/audit_cost_mcts_training.py \
  --data artifacts/diverse-cost-mcts-training-v1 \
  --out artifacts/diverse-cost-mcts-training-v1/partition-replay-audit.json
```

Если worker действительно завершится до completed, сначала проверить
живые процессы/ошибки/shard hashes. Затем допустим только явный `--resume`
с неизменным config/source/input/roots. Не создавать второй writer, не
менять TRAIN/holdout правило ради количества положительных примеров.

Следующий этап определяется полным отчётом допуска, не текущим малым
числом успехов. Формат намеренно несовместим со старым frozen-rollout
value loader: MCTS-prefix не имеет записанной stochastic policy probability,
а бюджетный неуспех не является абсолютной отрицательной меткой для хода.

## Обновление: сбор завершён, независимый аудит активен

2026-09-21. Collection завершилась с exit 0, все 256 условий и 32 shards
сохранены. Финальный manifest SHA256:
`3085862627fd4950d376dce04dbe727fd62a620dcaf028257322e8d296ae5aa5`.
Roots SHA256:
`8f1c14e9a85105b978d14ccd8654c071120ee5b0a3119fdc2b76e06547c35171`.
Всего 1494 продолжения, 113294 их хода, 114603 хода с tree-prefix,
35/256 solved conditions. Из 96 успешных продолжений 33 допущены по TRAIN
ownership; всего допущены 897 продолжений. Полностью TRAIN-owned условия —
35/256. Это результат TRAIN collection, не generalization score.

Независимый audit запущен указанной выше командой: unified session 39658,
PID 26574. На последнем срезе он replayed 72 условия / 418 episodes;
CPU продвинулся с 44.82 с до 2:15.61. Финального passed artifact пока нет,
данные не передавались оптимизатору. PID/эту запись не использовать как
доказательство текущей активности без новой проверки процесса/вывода.

Реализованы и проверены `mcts_training_supervision.py` и
`build_mcts_training_supervision.py`, контракт 55: set-valued positive actions,
success-only relative terms с exact prefix cancellation и explicit-unknot
suffix bounds, без фиктивных MCTS importance probabilities/value labels.
Последняя связанная серия: 78 tests passed; Ruff новых файлов чист.
Builder пока НЕ запускался: ждёт complete passed audit.
