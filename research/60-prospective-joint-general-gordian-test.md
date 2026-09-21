# Отдельный свежий TEST общих коротких Gordian-расстояний

2026-09-21. Заморожен до генерации и scoring. Дополняет short-u тест 58:
расстояние между двумя не обязательно тривиальными узлами не заменяется u(K).

Семейство — существующая replay-defined common-positive-block конструкция
`common_block_gordian.make_pair`. Seed=20261025, ровно 2048 attempts;
не продлевать недостаточную панель. Оба endpoint имеют signature lower bound
u>=3; query принадлежит TEST, reference не quarantine. Поэтому query-группы
не могут пересекаться с short-u<=2 queries протокола 58 по |signature|.
Не считать это независимостью от всякого когда-либо использованного узла:
явно проверить TRAIN paths и исключить все 13 прежних panel.json как в 58.

Прежние правила отбора без model scores: максимум четыре строки на
(query fingerprint, bound class), максимум 32 на класс. Классы: exact-0,
exact-1, exact-2 и interval. Completion gate: >=32 rows, >=8 query groups,
не менее четырёх rows в каждом exact классе 0/1/2. Порядок — по attempt index.
Failed/partial артефакт сохраняется, но не оценивается и не пополняется.

Выход: `artifacts/mcts-policy-metric-fresh-gordian-test-v1`. Все пути replayed;
upper — число CC до pair endpoint, не число CC полного пути до unknot.
Lower — signed-signature разность /2; при scoring дополнительно determinant
inequality, как в 58. Exact-2 показывает то, что невозможно увидеть только
по retrieval ranking. Нельзя превратить верхнюю границу в точное расстояние.

Дополнительный вид: прежний zero-CC lift query до N>=33/L>=512 с корректным
расширением pair-prefix; reference остаётся прежним. Это устойчивость
представления того же узла, а не самостоятельная задача большей сложности.

Сравнивать только замороженные parent, joint-1024 и metric-control-1024
протокола 57, плюс константы 0/1/2. Показать по обоим видам и точным классам
нарушения интервалов, равномерные по query fingerprint-группам агрегаты,
попарные различия. Общий короткий Gordian TEST дополняет, но не заменяет
full-policy MCTS L10/L1000 оценку по протоколу 58. Здесь не запускать
необъявленный поиск между двумя произвольными knots.

Это новая выборка query-групп из известного синтетического семейства,
а не доказательство обобщения на все семейства/размеры и не гарантия точных
Jones/Vassiliev invariants внутри embedding. Scoring только после complete
panel и passed lineage/independence audit. Никакого автоматического promotion.
