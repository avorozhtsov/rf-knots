from rf_knots.embedding_gordian import canonical_knot_id, parse_source_id


def test_source_names_parse_without_crossing_index_ambiguity() -> None:
    assert parse_source_id("K31m") == (("3_1", True),)
    assert parse_source_id("K10123") == (("10_123", False),)
    assert parse_source_id("K31K41") == (("3_1", False), ("4_1", False))


def test_only_exactly_representable_composites_are_mapped() -> None:
    assert canonical_knot_id("K31m") == ("knot:3_1", True)
    assert canonical_knot_id("K31K41") == ("knot:3_1#4_1", False)
    assert canonical_knot_id("K31mK41") is None
