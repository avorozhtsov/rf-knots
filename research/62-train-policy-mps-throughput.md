# TRAIN policy: совместимость MPS и реальная цена backward

2026-09-21. Изолированная инженерная проверка, пока неизменённый joint MCTS
проходит протокол 58. Тестовые узлы не используются. Нет optimizer.step,
новых checkpoint, выбора победителя по TEST или изменения live search.

## Контракт и данные

Готовый joint checkpoint; проверенные `diverse-mcts-supervision-diamonds-v1`
и certificate audit. Seed 20261028: выбрать без повторения 128 из 256
TRAIN-условий, внутри каждого равномерно один positive state; вложенные
batches 16/64/128. Это не 128 независимых knot types. Точное множество
допустимых positive actions сохраняется, единственный порядок witness
не навязывается. Три замера после одного warmup, один CPU thread.

Измеряется build_batch + полная policy сеть + set-valued positive loss +
backward + синхронизация device. **Не** включены metric loss, relative loss,
AdamW, загрузка/проверка данных и MCTS. После каждого arm проверены
неизменность всех weights/buffers и конечность градиентов/выходов.

На этом Mac 16 GiB RAM, PyTorch 2.13.0, MPS доступен;
`PYTORCH_ENABLE_MPS_FALLBACK` не задан. CPU/MPS сравнивались на одинаковых
данных/весах; старый и bulk adjacency — отдельно. Эти краткие concurrent
замеры не являются изолированной оценкой максимальной пропускной способности.

## Честное сохранение неудачного строгого gate

`mcts-policy-training-device-probe-v1` закончился
**failed_engineering_gate**: CPU совпал побитно, первый CPU/MPS numeric gate
прошёл, но original/bulk MPS gradients не совпали побитно. Первое найденное
отличие 2.98e-8 не является максимальным по всем параметрам.
Исходный report сохранён и не переименован в успешный.

Отдельный `mcts-policy-training-mps-repeatability-v1` повторил четыре backward
с теми же данными/весами для каждого helper. Даже неизменённый original MPS
давал разные gradients (76–77 tensors, max abs до 3.81e-6). Bulk повторения
и bulk-vs-original имели тот же порядок расхождений; integer adjacency на MPS
совпала точно. Поэтому битовое равенство обучения на MPS не обещается.

Только после этого запущен отдельный **numeric-v2**: CPU bit-exact gate
сохранён; для MPS использован тот же `rtol=1e-3, atol=1e-4`, который был
заранее объявлен в v1 для CPU/MPS. Допуск не выдаётся за побитовое равенство.
Все 12 arms прошли numerical/finite/unchanged-weight gates. На MPS original
сравнивается с самим собой в поле same-device comparator; повторяемость
доказывает/опровергает отдельный повторный probe, не этот тривиальный comparator.

## Результат: небольшое преимущество, не GPU-прорыв

Медианные секунды на policy forward/backward batch:

| Batch | CPU original | CPU bulk | MPS original | MPS bulk |
|---:|---:|---:|---:|---:|
| 16 | 0.0991 | 0.0883 | 0.1121 | 0.0974 |
| 64 | 0.3968 | 0.3634 | 0.4017 | 0.3751 |
| 128 | 0.8557 | 0.7848 | 0.7852 | 0.7211 |

CPU bulk: 181 / 176 / 163 policy-states/s, MPS bulk: 164 / 171 / 178.
Преимущество MPS bulk над CPU bulk на batch 128 — около 1.09×; на меньших
batch MPS медленнее. Ускорение encoder почти вдвое из заметки 61 **не**
перенеслось почти вдвое на весь policy backward: здесь другой workload,
включены action preparation, полный head и градиенты. Это повод отдельно
измерить стадии, а не предполагать, что облачная GPU автоматически решит
все ограничения потока данных. Nebius в этом probe не запускался.

Линейно пересчитывать эти числа в миллион независимых узлов или срок полного
RL нельзя: sample, посещённое состояние, neural evaluation, trajectory и
новый knot type — разные счётчики. Не исключены другие узкие места в
metric/relative losses, CPU data preparation, replay и sampling.

## Артефакты

- v1 failed probe SHA
  `bf97ac43fa79065eeaf2806e45b6225000691f94fc82847e996334f8ea6e8803`;
- repeated-gradient evidence SHA
  `8f17897d358be0b47ac90f047b1b49365827b42d12376e26d2beaeb49ac7384c`;
- numeric-v2 report SHA
  `f3e1f7ef85c8c7652af7b393b37bd0f344dfe21bedf2d0b0b714233c90564479`.

Каталоги: `artifacts/mcts-policy-training-device-probe-v1`,
`artifacts/mcts-policy-training-mps-repeatability-v1`,
`artifacts/mcts-policy-training-device-numeric-v2`. В каждом report.json
сохранены source/data hashes; v2 связан с обоими предшествующими отчётами.
Проверка policy gradients/CPU no-update покрыта двумя unit tests; GPU
численная совместимость подтверждена реальным запуском, не mock-тестом.
Frozen CPU evaluation продолжает работать без внедрения этих вариантов.
