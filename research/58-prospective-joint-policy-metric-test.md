# Новый TEST после обучения policy + metric

2026-09-21. Фиксируется до генерации и scoring, во время запуска протокола 57.
Ни completion gate, ни checkpoint не выбираются по качеству модели. Не менять
прежние панели и не называть повторное измерение на них независимым TEST.

## Короткое u и перенос размеров

Использовать существующий `build_constructive_short_u_panel.py`, seed=20261024,
ровно 4000 attempts, max_strands=24, min_depth=24, max_depth=80. По прежнему
правилу брать первые допустимые записи, максимум четыре на fingerprint-группу
и 32 на каждый класс числа CC witness (1 и 2). TEST-owner query, determinant
не равен 1; replay всех witnesses. Это синтетическое conjugate-block семейство,
а не случайная выборка всех узлов и не доказательство универсального переноса.

Исключить query-группы всех 13 уже существующих panel.json:
catalogue-independent, clean-paired-gordian, clean-paired-short-u,
common-block-gordian-supplement, constructive-short-u, diverse-metric-gordian,
diverse-metric-short-u, external-independent, fresh-group, general-gordian,
multisize-clean, short-u-heldout-panel, short-u-independent (все embedding-…-v1).
Оба endpoint прежних панелей участвуют в исключении; unknot reference нового
теста может быть известен. Недостаточная/failed прежняя панель тоже исключается.

Выход: `artifacts/mcts-policy-metric-fresh-short-u-test-v1`. Completion gate
существующего builder: минимум 32 представления, восемь fingerprint-групп,
восемь примеров с каждым CC upper bound. Не пополнять не прошедшую панель.
Fingerprint-группа не является точным knot ID; зеркала сгруппированы.

Основная evaluation_view — исходное представление. Отдельно показывать четыре
непересекающихся страты относительно TRAIN envelope L<=208/N<=17:
оба в envelope; только L>208; только N>17; оба больше. Не требовать заполнения
страт постфактум: пустая/малая страта означает отсутствие соответствующего
свидетельства. Дополнительный size_shift builder (N>=33, L>=256) — zero-CC lift
тех же узлов, то есть проверка устойчивости представления, а не новых типов.
Большие raw L/N не объявляются доказательством большой топологической сложности.

До scoring нужен независимый replay и проверка query isolation от ВСЕХ новых
supervision states и старых TRAIN paths, а не только от корней коллекции.
Краткие bounds: signature + determinant lower, verified-witness upper.
Группа exact-u2 отчётна отдельно: прошлое ухудшение калибровки нельзя скрыть
агрегированием с более многочисленными u<=1. Не заменять интервал его upper
bound как будто это точная метка.

## Сравнение замороженных checkpoint

Три learned arms: исходный actor (SHA 964199130db5645e41195905f8629636207fcf40e4bcf2a5683b9319fec7044e),
joint-1024 и metric-control-1024 по протокололу 57. Для расстояний дополнительно
константы 0, 1, 2. Показать нарушения интервалов, точные классы, равномерные
по fingerprint-группам агрегаты и попарные различия на тех же query.

Для поиска на всех validation query: λ=10 и 1000 отдельно, cost-MCTS с полными
policy continuations, восемь симуляций, horizon=192, четыре insertion candidates,
widening_batch=4, c_puct=1.5, temperature=1. Seed=20261026 + начало batch;
фиксированные batches по восемь условий (query-major, затем λ). Ресурсные caps
из reference witness через query_instance(witness_caps=True), одинаковые для
всех arms. Witness нужен только для общих caps и последующей проверки; его
действия не выдаются поиску. Uniform arm — одинаковый алгоритм со случайной
policy и равномерным tree prior. Value head не используется ни в одном arm.

Проверять каждое найденное свидетельство. Отдельно: число решений, новые и
потерянные решения, L10/L1000 на общих успехах и число улучшений/ухудшений;
не усреднять пропуски как нулевые стоимости. При одинаковом числе симуляций
compute неодинаков: учитывать neural states/batches, action scores, шаги
continuations, время. Eight-simulation результат не означает сходимость MCTS.

Эта панель проверяет u(K)=d(K,unknot), но не общий d(K1,K2). Свежая general
Gordian-панель потребует отдельного заранее фиксированного протокола; старую
common-block панель можно использовать только как явно повторный diagnostic.
Нет автоматического promotion, GPU-заказа, deployment, commit или push.
