# Сбор разнообразных TRAIN-траекторий для следующего joint-обучения

2026-09-21. Протокол до запуска; продолжение инженерного результата 52.
Это сбор данных, не новое обучение и не независимая оценка модели.

## Зафиксированные входы и выборка

- Corpus `embedding-diverse-short-u-corpus-v1/corpus.json`, SHA256
  `f3655e7dbfcc5b6ec7448b1f747dc2a4e1235fe036ab07fec61cc18df05c8f14`.
- Actor `embedding-diverse-metric-joint-v1/checkpoint.pt`, SHA256
  `964199130db5645e41195905f8629636207fcf40e4bcf2a5683b9319fec7044e`.
  Полная существующая сеть, strict loading, никаких новых весов/адаптеров.
- 128 различных TRAIN fingerprint-групп (det, |signature|). Из отсортированного
  списка групп равномерно выбрать без возвращения, затем равномерно одну
  исходную запись внутри каждой выбранной группы; Python Random(20261022).
  Не отбирать по оценке модели или успеху. Fingerprint не равен knot identity.
- Только исходные TRAIN записи, без новых lifts, отбрасывания трудных корней,
  добавления roots после результатов или подбора seeds. Если групп мало,
  остановиться, а не уменьшать заранее объявленный объём.
- Две условия для каждого корня: λ=10 и λ=1000. Всего 256 root/weight задач,
  но только 128 разных root representations. Начальные resource caps вмещают
  проверенный reference witness и фиксируются до inference. Reference CC —
  верхняя граница, не обязательно u. Pair interval корпуса не является u.

## Поиск и учёт вычислений

Для каждой задачи один cost-MCTS: 8 simulations, continuation horizon=128,
widening=4, c_puct=1.5; tree embedding-prior с τ=1 и stochastic full-policy
continuations с четырьмя insertion proposals по контракту 51. Окончание
поиска не при первом успехе. Все заявленные задачи выполнить независимо
от промежуточного качества.

Пакет — восемь последовательных root/weight задач (четыре корня). Search
seed=20261022+begin, где begin — индекс начала пакета. Внутренний seeding
зафиксирован исходником runner; learner probabilities не подменять
вероятностями MCTS-prefix. Сохранять prefix отдельно от rollout, полные
действия/состояния, вероятности выбранных rollout-ходов, остановки,
incumbents, verified witnesses, глубины, реальные neural/action counts.
Model и metric cache пересоздаются на каждом пакете: восстановление
незавершённого пакета не зависит от cache history предыдущих пакетов.

## Leakage и допуск к обучению

Каждый путь replay, включая tree-prefix. На корне и после каждого CC
вычислить точные determinant/signature; zero-CC ходы сохраняют их. Если
хотя бы одна группа не принадлежит TRAIN, всю траекторию сохранить, но
пометить ineligible. Дополнительно проверять пересечение с query-группами
старых corpus validation/size_shift и двух завершённых TEST panels:
`embedding-diverse-metric-short-u-test-v1/panel.json` и
`embedding-common-block-gordian-supplement-v1/panel.json`.

Для относительного обучения далее применять только условия, в которых
все записанные сравниваемые траектории прошли допуск; объявлять объём и
selection bias такого отбора. Не менять веса по данным до отдельного
аудита допущенных путей. Failure/horizon не становится абсолютным плохим
ходом. Пока не определён новый value-target, не подавать MCTS-результаты
в старый Monte-Carlo value-loader как будто это его frozen-policy episodes.

Новый формат намеренно отдельный: `rf-knots-cost-mcts-trajectories-v1`.
Старый `train_rollout_value.py` нельзя запускать на нём без корректного
нового контракта loss/proposals. Для policy нужны сопоставимые full-root
траектории с общим бюджетом и отменой общих префиксов; sampled rollout
probabilities не являются importance probabilities всего MCTS-пути.

## Восстановление и артефакты

Один writer lock. Manifest обновляется после выбора roots и после каждого
полностью replayed/annotated пакета, с SHA shard и счётчиками. При сбое
возобновлять только явно с теми же inputs/sources/config и выбранными roots.
Не менять завершённые shards; один незакоммиченный пакет можно повторить
детерминированно, сохранив и сверив его байты. Snapshot исходников сохраняется
до inference. Existing completed run не запускать снова.

После сбора: независимый replay/partition audit, описание разнообразия,
доля допущенных условий и полезных relative groups, затем заморозить joint
training + новый независимый TEST. Этот сбор не называется миллионом узлов
и не подтверждает перенос на произвольные L/N.
