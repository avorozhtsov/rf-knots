# Resource-conditioned rollout value and relative policy improvement

2026-09-20. Продолжение этапов E–F заметки 23. Глобальная цель неизменна:
размерно-независимая архитектура, короткие Gordian distances/u и хорошие L10/L1000.
Это один эксперимент policy evaluation/improvement, не завершённый большой RL.

## Определение value

`V_pi0(s,b) = P(одна полная случайная траектория фиксированной pi0 достигнет
empty one-braid в оставшихся ресурсных пределах b)`.

Здесь pi0 — закреплённая сеть и правило предложений действий. В каждый момент
предлагаются все O(L) обычные действия и до 16 INSERT из детерминированной
state-hash выборки. Это фиксированная rollout-политика, **не** полный progressive
MCTS catalogue. Действие выбирается случайно по её softmax. Лимит semantic moves
одновременно задаёт горизонт одной траектории; отдельного tree-search budget
у этой величины нет. Она не означает вероятность успеха MCTS за 512 симуляций.

Episode return: 1, если replay достиг unknot; 0 только при реальном исчерпании
горизонта или отсутствии legal proposals. Незавершённый rollout не получает 0.
Даже при существовании witness в тех же caps случайная политика может проиграть.
Такая метка не является математическим утверждением о неразвязываемости состояния
и не превращается в абсолютное «плохое действие». Pre-action states получают
Monte Carlo return этой **завершённой** траектории.

## Полный ресурсный вход, без residual adapter

В существующий SitePolicyValueNet добавлен opt-in budget_dim=5:
remaining moves, remaining CC, available length slack, available strand slack,
CC weight lambda (10 или 1000). До этого два последних геометрических ограничения
отсутствовали на входе, хотя меняли допустимые продолжения. Размеры весов не
зависят от максимальной длины или числа нитей.

Миграция checkpoint вставляет нулевые **колонки входа**, сохраняя исходные
policy logits и value в пределах точности float32. Нового выходного residual
блока нет; encoder, policy и value целиком обучаемы. Legacy budget_dim=2
остаётся default. Проверены точность миграции и построение resource context.

## Собранные данные

Parent — E/root checkpoint SHA
`7b03c5021a1c56cd65c0d48d0996b4735fd88ecd94897b74034763b2ac6714af`.
128 train suffix roots из 20 fingerprint groups, выборка по группам и глубине
suffix, без просмотра прогнозов. Каждая четвёртая запись дополнительно поднята
zero-CC witness до 9 нитей/60 букв. Ресурсные caps включают предоставленный
suffix witness с запасом восьми semantic moves. Внутри группы все 8 seeds
получают строго одинаковые root/caps/lambda.

`artifacts/embedding-rollout-value-corpus-v1`: 1024 завершённых эпизода,
37 798 переходов, 342 успеха. 733 эпизода допустимы для train, из них 312 успешны.
Любой заход в не-train fingerprint group исключает **весь** эпизод из обучения;
эпизод и причина исключения сохраняются. Это фильтр утечки, но он может смещать
распределение: успех среди принятых отличается от успеха всех эпизодов.
Нельзя объявлять полученный value откалиброванной вероятностью без проверки
на новых траекториях, включая исходы, которые были бы исключены из train.

SHA данных:

- manifest: `d7ac313b6a6756cbfc3b46f3627ce7eab8ddc1fe85d1647c7ba2ef968c528a6a`;
- actor: `597a205c5077a28943e887f73cecf45f4d7b203fe1b3b1db30ffa7c8674b5335`;
- roots: `c428f341fdc86911763a722fd91c25605ce5d59d0861e6ef2eac6636ba44dfd9`.

Коллектор сохраняет immutable episode shards, source snapshots и checkpoint
курсора между группами. Resume проверяет config/inputs/source hashes. Только
полные shards коммитятся; преждевременно прерванный chunk повторяется с теми же
seeds, не превращаясь в неуспех.

## Targets и контролируемое обучение

Seeds 0–5 используются для обучения, seeds 6–7 — только для диагностической
проверки rollout value. Это **те же train roots**, а не независимые типы узлов.
В проверке участвуют все 256 исходов seeds 6–7, включая траектории через
зарезервированные группы; их состояния/метки не передаются optimizer.

Value BCE видит 19 194 pre-action строки допустимых train episodes. Строки
семплируются равномерно по посещениям, без отдельного деления на длину траектории.
Constant baseline — success rate train root episodes, не подогнанный по оценке.

Для policy берутся только группы, где все шесть train эпизодов допустимы.
Есть 38 групп с сигналом предпочтения. Лучший реальный L_lambda среди успешных
путей получает суммарный +1 (делится между ties), строго худшие исходы — суммарный
-1. Если все неуспешны или все равны, policy target отсутствует. Неизвестную
стоимость неуспешного пути не заменяем фиктивным числом.

Коэффициенты агрегируются по (state с оставшимся бюджетом, action) **точными
рациональными числами до вычисления loss**. Общий префикс +1 и девяти -1/9
исчезает точно; длины путей не нормируются по отдельности. Поздний возврат в то же
слово с меньшим остатком ходов — другой context, он не обязан сокращаться.

Для стабильности фиксированных данных используется clipped importance-ratio
surrogate с clip=0,2 относительно pi0. Это модификация относительной policy-loss,
не заявление об использовании vanilla PPO/GRPO или о гарантии монотонного
улучшения. Основание механизма clipping:
[Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
Сокращение префиксов выполняется **до** нелинейного clipping. Сохраняются доля
clipped terms и средний абсолютный log-ratio; clipping не ограничивает строго
сдвиг общей сети под действием других losses.

Два matched arm, одинаковый parent, RNG и batches:

1. Metric + 0,1*relative-policy + value BCE.
2. Metric + 0,1*relative-policy, value weight=0.

2048 updates, batch64 для metric и value, одна группа для relative-policy,
AdamW lr=2,5e-5, norm clipping=1, CPU. Metric batch — 63 uniform train pair rows
из frozen augmented corpus и mirror-trefoil sentinel. Это новый протокол sampler,
поэтому причинный контроль — matched arm, а не только сравнение с E.
Value обучается оценивать pi0, хотя обновлённая policy становится pi1. Для
следующей итерации нужны новые rollouts pi1; старые labels нельзя переименовать
в измерение pi1.

16-step preflight прошёл, без nonfinite: Brier 0,2513 -> 0,2413, constant=0,2298.
Эти цифры — инженерная диагностика на тех же train roots; по ним не делаем
научного promotion. Полные arm должны закончиться прежде, чем их оценивать.

## Проверка после обучения

Проверить checkpoint SHA, совпадение начальных моделей/RNG/batches, шаги,
изменение shared encoder и сохранение chirality/short-distance metrics.
Сравнить rollout Brier с constant baseline, не только с исходной сетью.
На development knot-heldout panel выполнить одинаковый sparse MCTS:
learned value против constant 0,5 у того же checkpoint и matched arm без value.
Все найденные пути replay-проверяются; failures остаются censored, L10/L1000
first-found не называются оптимальными. Старую уже просмотренную независимую
test-панель нельзя использовать для настройки и потом объявлять новой.

## Фактический запуск

Основные arm запущены как `embedding-rollout-joint-value-v1` (PID 44569) и
`embedding-rollout-joint-control-v1` (PID 44582). При первой проверке оба прошли
32/2048 обновления; CPU time и checkpoint/manifest реально продвигались.
`before.json` одинаковы: SHA
`9eb6d13b5d45eec4ffc01ee3c37fd92df7652937922a6662c2764f660b09f26f`.
Старые PID не являются последующим доказательством жизни: при продолжении
проверять процессы, шаги и свежесть файлов заново. Все исходные данные и source
snapshots сохранены; завершение этих arm пока не заявляется.

## Completion and independent check

Оба arm действительно завершили `2048/2048`, без живых процессов. Финальные SHA:

- value: `9d281291d0e1a1d088a274aa26060e5b3210e51d82d5db25e76871d43d4fbbd4`;
- matched control: `33ac6f3036cf0a02d57599bea6ccb4a1fea1508c6b616c88bac5d6a1bf28edb2`.

Inputs и source hashes совпали; RNG states совпали на одинаковых шагах. Shared
model states после обучения различаются.

На same-root rollout diagnostic value-arm снизил Brier с `0.25126` до `0.17736`,
но constant baseline даёт `0.22977`; control остался `0.25124`. Metric interval
error стал `0.34982` у value и `0.35377` у control (до обоих `0.36210`). Это
не knot-disjoint evidence: seeds 6–7 новые, но roots те же.

Внешняя панель содержит 11 roots в трёх коррелированных группах, из них 10
exact-u2. Metric interval violation: value `0.46916`, control `0.44892`;
exact-u2 MAE `0.51608` / `0.49381`. Поэтому value training не дал независимого
улучшения расстояния.

Sparse MCTS (3 groups x 4 seeds x 512, widening batch 4, same witness caps):

| arm | leaf value | solved trees |
| --- | --- | ---: |
| value | learned | 1/12 |
| value | constant 0.5 | 0/12 |
| control | learned | 0/12 |
| control | constant 0.5 | 1/12 |

Результат взаимно компенсируется на малой панели и не поддерживает claim, что
learned value улучшил MCTS.

Anytime cost-search, 64 expansions, решил 8/11 у обоих arm. Value проиграл на
одном общем решении: `short-u-2` получил L10/L1000 `22/1012`, control — `19/1009`;
новых решений не было. Failures остаются censored.

Value-arm не получает promotion. Следующий шаг — новые knot-disjoint roots и
новые rollouts уже обновлённой pi1; старые pi0 labels нельзя переименовать в
данные pi1. Для value нужна отдельная калибровка на новых fingerprint groups.

## Следующая итерация pi1

После завершения value-arm собран отдельный
`embedding-rollout-value-pi1-corpus-v1` с actor SHA
`322a55424347b30da2ff08a016927cffd77d99c2908f7f180d9ccbcf6b426fd4`:
1024 episodes, 37 593 transitions, 373 successes, 755 eligible episodes,
340 eligible successes. Root SHA остался тем же (`c428...dfd9`), поэтому это
новая policy-lineage на тех же train groups, не независимый test.

Запущен новый 1024-step training `embedding-rollout-joint-pi2-v1` на этих pi1
rollouts. Его checkpoint и before/after metrics будут проверены отдельно; нельзя
считать launch результатом до финального SHA и matched evaluation.

## Pi2 completion and independent evaluation

The pi2 continuation completed normally at 1024/1024 steps.  The terminal
manifest is `completed`, there is no surviving training process, and the
checkpoint hash agrees with the manifest:

`embedding-rollout-joint-pi2-v1/checkpoint.pt` SHA-256
`18a1b45cab164d0d47225447731fe9f282f9378a1edeb86afe4a5270f17b084f`.

Its frozen input hashes point to the separate pi1 corpus
(`embedding-rollout-value-pi1-corpus-v1`, actor SHA
`322a55424347b30da2ff08a016927cffd77d99c2908f7f180d9ccbcf6b426fd4`) and the
same short-u metric corpus as the earlier arm.  The run took 729.27 seconds;
the final logged losses were value 0.48597, policy 0.27113, metric 0.01929.

The matched development validation moved only slightly: mean interval
violation `0.34982 -> 0.35435` for pi2.  Exact-group MAE changed from
`0.31576/0.46300/0.53883` to `0.31440/0.46610/0.56875` for groups 0/1/2.
This is not a promotion on the development metric.

The frozen external panel was evaluated without changing the checkpoint or
panel.  It contains 11 correlated roots (10 exact-u2):

| checkpoint | mean interval violation | exact-u2 MAE |
| --- | ---: | ---: |
| pi1 (`embedding-rollout-joint-value-v1`) | 0.46916 | 0.51608 |
| pi2 | 0.47281 | 0.52009 |

The immutable reports are `embedding-rollout-joint-pi2-v1/external-panel.json`
and `external-search.json`; the corresponding pi1 reports are stored beside
its checkpoint.  Thus pi2 is slightly worse on this independent distance panel.

Matched sparse MCTS used 12 trees (3 fingerprint groups × 4 seeds), 512
simulations, witness caps and widening batch 4.  Learned-value search solved
pi2 `2/12` trees versus pi1 `1/12`; policy-matched constant-0.5 controls solved
`0/12` for both.  This is a weak positive search signal, not a statistically
reliable promotion: the panel is tiny, correlated, and failures are censored.

The 64-expansion anytime cost search over all 11 queries solved `8/11` for
both checkpoints.  Among solved queries, pi2 mean costs were `L10=26.25` and
`L1000=1882.5`, versus pi1 `26.625` and `1882.875`; the difference comes from
one shared short-u query (`19/1009` for pi2 versus `22/1012` for pi1).  These
are first-found costs under equal expansion count, not optimal distances, and
the two searches used different numbers of neural evaluations internally.

Conclusion: pi2 is a reproducible completed training arm with a small,
non-conclusive search improvement but no independent distance improvement.
It does not justify a GPU promotion or a claim that the embedding is useful in
general.  The next scientifically useful step is a larger knot-disjoint pi1
rollout corpus and matched seeds, while the architecture work proceeds toward
hierarchical windows and factorized action selection for large `L` and `N`.

## Pi2 actor rollouts and pi3 continuation

To avoid relabelling pi0 data, a new corpus was collected from the completed pi2
checkpoint using the same frozen train partition, `u_views.json`, and a new
collection seed (`20260927`).  The corpus is
`embedding-rollout-value-pi2-corpus-v1`: 1024 episodes, 37,114 transitions,
436 successes, 743 eligible episodes, and 388 eligible successes.  Its manifest,
actor, and roots SHA-256 values are respectively
`4d7619135f49d77ace2711b248f776618b8c9463efa7f7a9e367631d53ced99c`,
`e8e38b685e15f8ed6457fadda7309733557a60dcf27273003525cb8763782da2`, and
`4805e54e792bb8a3d404e153401b42f08b9cffb3c25b724c3eef0a0558c66234`.
The actor input checkpoint SHA is the verified pi2 SHA above.  The corpus is a
new policy lineage, but not a new knot partition; that limitation remains
explicit.

Training `embedding-rollout-joint-pi3-v1` has now been launched on this corpus
for 1024 steps with the frozen metric corpus and the same resource-conditioned
objective.  At the first live check its manifest was `training`, `16/1024`, and
the checkpoint was fresh.  No promotion claim is made until it finishes and is
evaluated on the frozen external panel and matched search protocol.

## Pi3 completion and comparison with pi2

The pi3 continuation completed at 1024/1024 steps with no surviving training
process. Its checkpoint SHA-256 is
`9832e11da637256a73e33c0afc623a9609d3f14b96bb4b816628c8c7b88fcde9`, matching
the terminal manifest. The run took 595.21 seconds. Its input hashes match the
completed pi2-rollout corpus, actor SHA
`e8e38b685e15f8ed6457fadda7309733557a60dcf27273003525cb8763782da2`, and the
frozen short-u metric corpus.

On matched development validation, pi3 changed mean interval violation from
pi2 `0.354353` to `0.353192`. Exact-group MAE changed from
`0.314404/0.466103/0.568754` to `0.319363/0.445475/0.563109` for groups
0/1/2.

On the frozen external panel (11 correlated roots, 10 exact-u2), pi3 improved
over both earlier arms:

| checkpoint | mean interval violation | exact-u2 MAE |
| --- | ---: | ---: |
| pi1 | 0.46916 | 0.51608 |
| pi2 | 0.47281 | 0.52009 |
| pi3 | 0.46579 | 0.51237 |

Matched sparse MCTS used 12 trees, 512 simulations, witness caps and widening
batch 4. Pi3 learned-value search solved `2/12` trees, while its constant-0.5
control solved `1/12`. This remains too small and correlated for a promotion
claim; failures are censored and witnesses are first-found, not cost-optimal.

The 64-expansion anytime cost search solved `8/11` for pi3, with mean
`L10=26.25` and `L1000=1882.5` among solved queries. This equals pi2's cost
portfolio; pi3 did not yet produce a new solve or a shorter returned path.
The immutable reports are under `embedding-rollout-joint-pi3-v1/` and include
checkpoint/panel hashes and source snapshots.

Conclusion: pi3 is the first continuation with a small consistent improvement
on the external distance panel, but search-cost improvement is not shown. The
next useful step is a genuinely knot-disjoint rollout cohort before any large
GPU promotion; repeated same-partition self-play alone is insufficient evidence
of generalization.

## Pi4 continuation on the v2 corpus

The pi4 continuation was trained from the verified pi3 checkpoint policy on
`embedding-rollout-value-pi3-corpus-v2`, using the v2 short-u metric corpus.
The run completed normally at 1024/1024 steps in 647.66 seconds.  The terminal
manifest is `completed`; the checkpoint SHA-256 is
`283235ed605b638cd7a7bf68159b997f98880cc2acba662e44ac10c3eff13f6f`, matching
the bytes on disk.  Final logged losses were metric `0.1605808` and value
`0.4202915`.

This v2 corpus is a new root/policy distribution, but the strict audit found
107 overlapping canonical path fingerprints with the earlier v1 train corpus;
literal root strings happened not to overlap.  It must therefore not be called
a knot-disjoint test or a generalization proof.

On the frozen external panel (11 correlated roots, 10 exact-u2), pi4 regressed:

| checkpoint | mean interval violation | exact-u2 MAE |
| --- | ---: | ---: |
| pi3 | 0.46579 | 0.51237 |
| pi4 | 0.52601 | 0.57861 |

Matched sparse MCTS (12 trees, 512 simulations, witness caps, widening batch 4)
solved `1/12` trees with the learned value and `1/12` with the constant-0.5
control.  The 64-expansion anytime cost search solved `8/11` queries; among
solved rows its mean costs were `L10=26.625` and `L1000=1882.875`, versus pi3's
`26.25` and `1882.5`.  These remain first-found costs, not optimal distances,
and failures are censored.

Conclusion: pi4 is a valid completed lineage and a negative promotion result
for this data mixture.  More same-partition self-play or a GPU scale-up is not
the next justified action.  The next architecture step is to make long-word,
many-strand inputs computationally feasible and to build a genuinely
fingerprint-disjoint evaluation before another large training budget.

## Hierarchical-window architecture pilot

The first opt-in hierarchical-window trunk was trained for 1024 steps on the
same pi3-v2 rollout lineage.  It used `window_size=8` and four shared-weight
relative-window levels; the ten new window parameters were random-initialized
and are recorded in the manifest.  The run completed in 674.37 seconds with
checkpoint SHA-256
`b59387903d80e3ac0a0fa17dcd47a442664965ef5d9a5533a1df2a9ef4f13f71`.

On the matched development panel its mean interval violation was `0.39998` and
exact-group MAE was `0.35439/0.45124/0.64765` for groups 0/1/2.  On the frozen
external panel it was `0.52929` with exact-u2 MAE `0.58222`, worse than pi3 and
pi4.  Learned-value MCTS solved `0/12` trees; the constant control solved
`1/12`.  The 64-expansion cost search solved `8/11`, with mean `L10=26.25` and
`L1000=1882.5`, equal to pi3 and slightly better than pi4 by the shared
first-found-cost statistic.

This is a valid feasibility/training result, not evidence for promotion: the
external panel is tiny and correlated, and the v2 training corpus is not strict
knot-disjoint.  It does show that the hierarchical trunk can be initialized
from the old checkpoint, trained end-to-end, and evaluated under the same
replay contract without a fixed `L/N` parameter table.

## Factorized proposal-head replay pilot

The opt-in `FactorizedSiteHead` was trained with the encoder, historical
SiteHead, and value head frozen.  The training rows were the 57 fixed-root
relative-policy groups from `embedding-rollout-value-pi3-corpus-v2` (9,747
replayed terms).  Only scores for the frozen behavioural proposal set entered
the denominator; unopen/unseen actions were not negative labels.

The first run (`...factorized-proposal-pilot-v2`, SHA
`33b694acf1683b24e668ca136594f85e82c07b8157b03325f294986529cce45a`) used an
unstable sign-incorrect clipping rule and produced losses as large as millions;
its unchanged recall is retained as a failure diagnostic, not a result.

The matched corrected run (`...factorized-proposal-pilot-v4`) used the same
seed, a sign-correct clipped relative objective and `lr=1e-4`.  Its checkpoint
SHA is `87df92de99d8f4afe7a4bb61c63c3241c765ef23eae72421fe87458b3f62e619`.
On 512 heldout-seed behavior actions from the **same roots**, top-8/top-8
proposal recall moved from `0.64258` to `0.66211`; mean proposed actions were
`48.82` and `48.89`.  This is a useful head-training sanity signal, not an
independent topology test or a search-quality result: behavior actions are not
optimal, roots overlap the training partition, and MCTS has not yet consumed
these proposals.

## Factorized proposals in sparse MCTS

The v4 checkpoint was then evaluated on the frozen external panel with the
new factorized proposal provider.  The protocol used 3 selected instances x 4
seeds, 512 simulations, witness caps, widening batch 4, learned leaf value,
and `cc_weight=1000`.  The immutable report is
`artifacts/embedding-factorized-external-search-v1/report.json`; checkpoint
SHA is `87df92de99d8f4afe7a4bb61c63c3241c765ef23eae72421fe87458b3f62e619` and
panel SHA is `b8618f4005e6f889128c30aa3504b48ac55f07465c16955f2ca544775fe5636a`.
It solved 4/12 trees, all four seeds of `short-u-1`, with first-found costs
L10=27 and L1000=2007.  It solved 0/8 trees for `short-u-2` and `short-u-7`.

A matched report with the same checkpoint, panel, seeds, simulations, caps,
and widening but without the factorized provider is
`artifacts/embedding-standard-external-search-v1/report.json`.  It solved
0/12 trees, including 0/4 on `short-u-1`.  The factorized run also made more
neural action scores (51,845 versus 34,255), so the comparison is a search
ordering/compute trade-off rather than a pure equal-cost policy comparison.

This is the first evidence that a trained factorized proposal head can change
which sparse branches are opened and recover a short witness under the same
simulation cap.  It is not yet evidence of topology-level generalization:
the panel is only 11 correlated roots, the proposal head was replay-trained
from overlapping roots, and the two harder families remain unsolved.  Future
promotion must use a fingerprint-disjoint panel and report success versus
neural-evaluation budget, with a random/fixed-proposal control.

## Size-shift execution contract

The search evaluator now has an explicit `--panel-view size_shift` mode.  It
uses the frozen panel's provenance-preserving larger representations rather
than silently treating them as ordinary validation rows.  On the v4 checkpoint
(60 word positions, 9 strands; 3 selected roots x 2 seeds x 256 simulations),
factorized proposals solved 0/6 and the matched ordinary widening solved 0/6.
The reports are `artifacts/embedding-factorized-size-shift-v1/report.json` and
`artifacts/embedding-standard-size-shift-v1/report.json`.

This is a successful size-independent execution smoke: fixed-size parameters
did not block a 60/9 input and the exact legality/catalogue fallback remained
live.  It is not a quality result; both arms failed under a small budget and
the size-shift rows are lifted, correlated witnesses rather than new knot
types.  A real promotion still needs larger fingerprint-disjoint roots and a
budget sweep.
