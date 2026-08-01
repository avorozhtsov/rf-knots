"""Catalogue identifiers and provenance are part of the persisted data contract."""

from rf_knots.knot_table import canonical_name, load_table, lookup, unknotting_numbers


def test_knotinfo_names_are_canonical_and_spherogram_names_remain_aliases():
    canonical = lookup("12n_570")
    assert canonical is not None
    assert lookup("K12n570") == canonical
    assert canonical_name("12n_570") == "12n_570"
    assert canonical_name("K12n570") == "12n_570"
    assert canonical_name("K12n121") == "12n_121"  # recorded strand-cap skip
    assert canonical_name("not-a-knot") is None


def test_knotinfo_correspondence_is_complete_and_provenance_bearing():
    table = load_table()
    sources = table["identifier_sources"]
    tabulated_names = set(table["knots"]) - {"0_1"}
    skipped_names = {entry["name"] for entry in table["skipped"]}

    assert table["schema_version"] == 2
    assert table["canonical_names"] == "KnotInfo"
    assert set(sources) == tabulated_names | skipped_names
    assert all(names["KnotInfo"] == canonical for canonical, names in sources.items())
    assert len({names["Spherogram"] for names in sources.values()}) == len(sources)
    assert sources["12n_647"] == {"KnotInfo": "12n_647", "Spherogram": "K12n647"}
    assert table["catalogue_provenance"] == {
        "database_url": "https://knotinfo.org/knotinfo_data_complete.xls",
        "name": "KnotInfo",
        "retrieved_at": "2026-08-01",
        "scope": ("identifier correspondence only; braid representatives come from "
                  "Spherogram and invariants are computed by rf-knots"),
        "sha256": "bd454dcb6bcd5effe205b27ca9de172bb21cf87ce190e15f870e1b07a714ccbe",
        "url": "https://knotinfo.org/",
    }


def test_current_knotinfo_unknotting_numbers_use_canonical_names():
    values = unknotting_numbers()["values"]
    assert values["12n_570"] == 2
    assert values["12n_647"] == 4
