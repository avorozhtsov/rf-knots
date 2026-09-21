# Joint policy/metric после MCTS и проверенных альтернатив

2026-09-21. Протокол до обновления весов. Фиксированная пара запусков, не
million-scale RL и не итоговая оценка произвольных L/N.

Вход — completed `diverse-mcts-supervision-diamonds-v1` и passed
`certificate-audit.json`, плюс исходный metric corpus с SHA
`f3655e7dbfcc5b6ec7448b1f747dc2a4e1235fe036ab07fec61cc18df05c8f14`.
Начальные веса обеих веток — actor collection 53, SHA
`964199130db5645e41195905f8629636207fcf40e4bcf2a5683b9319fec7044e`.
Проверить транзитивные input hashes, все target shards, replay источников
short-u views. Архитектура и размеры весов не меняются; новых heads нет.

## Две ветки

1. `joint`: metric + 0.1 * set-valued policy + 0.01 * relative-success loss.
2. `metric-control`: только та же metric loss, с теми же входными draw/seed
   и числом optimizer steps. Неиспользованные policy draws тоже выполняются,
   чтобы не разойтись по random stream. Compute при этом не одинаковый;
   отдельно записывать время и policy/metric samples.

Общий encoder и policy head доступны joint-оптимизатору; scalar value head
заморожена, потому что эти данные не определяют для неё корректную вероятность
успеха. Encoder при обучении меняется, поэтому даже численные value outputs
могут измениться; они не калибруются и не используются как MCTS returns.
Это не head-only residual adapter. В control encoder тоже меняет policy
outputs косвенно, несмотря на отсутствие policy loss.

## Фиксированные настройки

1024 шага, seed=20261023, CPU/один thread, AdamW lr=2.5e-5,
weight_decay=0.01, gradient norm clip=1.0. Сохранять progress каждые 16 шагов,
RNG и optimizer; не выбирать checkpoint по TEST и не останавливаться по loss.

На шаг: 31 sample из прежнего group-balanced metric TRAIN, 32 из нового
explicit-unknot suffix TRAIN (равномерно bound class → fingerprint → row),
плюс прежний фиксированный exact mirror-distance-2 sentinel. Signature и
determinant bounds в прежних rows по тому же контракту, что эксперимент 42.

Policy: 32 samples, равномерно condition, затем state внутри condition.
Положительная loss — −log массы всего доказанного positive set. Relative
group выбирается равномерно из допустимых групп, сумма после точной отмены
общих state/actions без нормировки отдельных траекторий по длине.
Clipped ratio с clip=0.2 использует reference probabilities исходной сети
на фиксированных training pools, а не rollout/MCTS behaviour probabilities.
Это proximal ограничение supervised preferences, не заявление о PPO/GRPO.

## Ограничения данных и заключений

Сбор дал 128 root fingerprint-групп и 256 условий, 1494 траектории; не
объявлять sampled training rows или 94362 добавленных action certificates
новыми независимыми узлами. Без proof augmentation было лишь четыре
multi-positive состояния из 19189. После него заявлены 8902; перед train
обязателен полный certificate replay. Positive sets всё ещё не исчерпывают
всех решений. Relative supervision мала: четыре условия / 514 terms после
добавления reference witnesses. Её вклад не выдавать за масштабное self-play.

Ни failure labels, ни scalar-value loss не добавлять для искусственного баланса
50/50. Все изменения целей обоснованы существующими успешными witnesses,
а не критериями «не нашёл при малом бюджете».

После обеих веток — проверка tensor/checkpoint hashes и завершённости.
Для научного вывода нужен отдельный новый, замороженный до scoring TEST:
короткие Gordian/u интервалы с constant controls и replayed L10/L1000 поиска,
с общими корнями, ресурсами и явным учётом compute. Старые известные панели
не считать свежим promotion TEST. Автоматического promotion/deploy/push нет.
