import pytest
from thinc.api import NumpyOps, get_current_ops

import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example


@pytest.fixture
@registry.misc("span_ruler_patterns")
def patterns():
    return [
        {"label": "HELLO", "pattern": "hello world", "id": "hello1"},
        {"label": "BYE", "pattern": [{"LOWER": "bye"}, {"LOWER": "bye"}]},
        {"label": "HELLO", "pattern": [{"ORTH": "HELLO"}], "id": "hello2"},
        {"label": "COMPLEX", "pattern": [{"ORTH": "foo", "OP": "*"}]},
        {"label": "TECH_ORG", "pattern": "Apple"},
        {"label": "TECH_ORG", "pattern": "Microsoft"},
    ]


@pytest.fixture
def overlapping_patterns():
    return [
        {"label": "FOOBAR", "pattern": "foo bar"},
        {"label": "BARBAZ", "pattern": "bar baz"},
    ]


@pytest.fixture
def person_org_patterns():
    return [
        {"label": "PERSON", "pattern": "Dina"},
        {"label": "ORG", "pattern": "ACME"},
        {"label": "ORG", "pattern": "ACM"},
    ]


@pytest.fixture
def person_org_date_patterns(person_org_patterns):
    return person_org_patterns + [{"label": "DATE", "pattern": "June 14th"}]


def test_span_ruler_add_empty(patterns):
    """Test that patterns don't get added excessively."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler", config={"validate": True})
    ruler.add_patterns(patterns)
    pattern_count = sum(len(mm) for mm in ruler.matcher._patterns.values())
    assert pattern_count > 0
    ruler.add_patterns([])
    after_count = sum(len(mm) for mm in ruler.matcher._patterns.values())
    assert after_count == pattern_count


def test_span_ruler_init(patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(patterns)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 4
    assert "HELLO" in ruler
    assert "BYE" in ruler
    doc = nlp("hello world bye bye")
    assert len(doc.spans["ruler"]) == 2
    assert doc.spans["ruler"][0].label_ == "HELLO"
    assert doc.spans["ruler"][0].id_ == "hello1"
    assert doc.spans["ruler"][1].label_ == "BYE"
    assert doc.spans["ruler"][1].id_ == ""


def test_span_ruler_no_patterns_warns():
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    assert len(ruler) == 0
    assert len(ruler.labels) == 0
    assert nlp.pipe_names == ["span_ruler"]
    with pytest.warns(UserWarning):
        doc = nlp("hello world bye bye")
    assert len(doc.spans["ruler"]) == 0


def test_span_ruler_init_patterns(patterns):
    # initialize with patterns
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    assert len(ruler.labels) == 0
    ruler.initialize(lambda: [], patterns=patterns)
    assert len(ruler.labels) == 4
    doc = nlp("hello world bye bye")
    assert doc.spans["ruler"][0].label_ == "HELLO"
    assert doc.spans["ruler"][1].label_ == "BYE"
    nlp.remove_pipe("span_ruler")
    # initialize with patterns from misc registry
    nlp.config["initialize"]["components"]["span_ruler"] = {
        "patterns": {"@misc": "span_ruler_patterns"}
    }
    ruler = nlp.add_pipe("span_ruler")
    assert len(ruler.labels) == 0
    nlp.initialize()
    assert len(ruler.labels) == 4
    doc = nlp("hello world bye bye")
    assert doc.spans["ruler"][0].label_ == "HELLO"
    assert doc.spans["ruler"][1].label_ == "BYE"


def test_span_ruler_init_clear(patterns):
    """Test that initialization clears patterns."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(patterns)
    assert len(ruler.labels) == 4
    ruler.initialize(lambda: [])
    assert len(ruler.labels) == 0


def test_span_ruler_clear(patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(patterns)
    assert len(ruler.labels) == 4
    doc = nlp("hello world")
    assert len(doc.spans["ruler"]) == 1
    ruler.clear()
    assert len(ruler.labels) == 0
    with pytest.warns(UserWarning):
        doc = nlp("hello world")
    assert len(doc.spans["ruler"]) == 0


def test_span_ruler_existing(patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler", config={"overwrite": False})
    ruler.add_patterns(patterns)
    doc = nlp.make_doc("OH HELLO WORLD bye bye")
    doc.spans["ruler"] = [doc[0:2]]
    doc = nlp(doc)
    assert len(doc.spans["ruler"]) == 3
    assert doc.spans["ruler"][0] == doc[0:2]
    assert doc.spans["ruler"][1].label_ == "HELLO"
    assert doc.spans["ruler"][1].id_ == "hello2"
    assert doc.spans["ruler"][2].label_ == "BYE"
    assert doc.spans["ruler"][2].id_ == ""


def test_span_ruler_existing_overwrite(patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler", config={"overwrite": True})
    ruler.add_patterns(patterns)
    doc = nlp.make_doc("OH HELLO WORLD bye bye")
    doc.spans["ruler"] = [doc[0:2]]
    doc = nlp(doc)
    assert len(doc.spans["ruler"]) == 2
    assert doc.spans["ruler"][0].label_ == "HELLO"
    assert doc.spans["ruler"][0].text == "HELLO"
    assert doc.spans["ruler"][1].label_ == "BYE"


def test_span_ruler_serialize_bytes(patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(patterns)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 4
    ruler_bytes = ruler.to_bytes()
    new_nlp = spacy.blank("xx")
    new_ruler = new_nlp.add_pipe("span_ruler")
    assert len(new_ruler) == 0
    assert len(new_ruler.labels) == 0
    new_ruler = new_ruler.from_bytes(ruler_bytes)
    assert len(new_ruler) == len(patterns)
    assert len(new_ruler.labels) == 4
    assert len(new_ruler.patterns) == len(ruler.patterns)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert sorted(new_ruler.labels) == sorted(ruler.labels)


def test_span_ruler_validate():
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    validated_ruler = nlp.add_pipe(
        "span_ruler", name="validated_span_ruler", config={"validate": True}
    )

    valid_pattern = {"label": "HELLO", "pattern": [{"LOWER": "HELLO"}]}
    invalid_pattern = {"label": "HELLO", "pattern": [{"ASDF": "HELLO"}]}

    # invalid pattern raises error without validate
    with pytest.raises(ValueError):
        ruler.add_patterns([invalid_pattern])

    # valid pattern is added without errors with validate
    validated_ruler.add_patterns([valid_pattern])

    # invalid pattern raises error with validate
    with pytest.raises(MatchPatternError):
        validated_ruler.add_patterns([invalid_pattern])


def test_span_ruler_properties(patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler", config={"overwrite": True})
    ruler.add_patterns(patterns)
    assert sorted(ruler.labels) == sorted(set([p["label"] for p in patterns]))


def test_span_ruler_overlapping_spans(overlapping_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(overlapping_patterns)
    doc = ruler(nlp.make_doc("foo bar baz"))
    assert len(doc.spans["ruler"]) == 2
    assert doc.spans["ruler"][0].label_ == "FOOBAR"
    assert doc.spans["ruler"][1].label_ == "BARBAZ"


def test_span_ruler_scorer(overlapping_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(overlapping_patterns)
    text = "foo bar baz"
    pred_doc = ruler(nlp.make_doc(text))
    assert len(pred_doc.spans["ruler"]) == 2
    assert pred_doc.spans["ruler"][0].label_ == "FOOBAR"
    assert pred_doc.spans["ruler"][1].label_ == "BARBAZ"

    ref_doc = nlp.make_doc(text)
    ref_doc.spans["ruler"] = [Span(ref_doc, 0, 2, label="FOOBAR")]
    scores = nlp.evaluate([Example(pred_doc, ref_doc)])
    assert scores["spans_ruler_p"] == 0.5
    assert scores["spans_ruler_r"] == 1.0


@pytest.mark.parametrize("n_process", [1, 2])
def test_span_ruler_multiprocessing(n_process):
    if isinstance(get_current_ops, NumpyOps) or n_process < 2:
        texts = ["I enjoy eating Pizza Hut pizza."]

        patterns = [{"label": "FASTFOOD", "pattern": "Pizza Hut"}]

        nlp = spacy.blank("xx")
        ruler = nlp.add_pipe("span_ruler")
        ruler.add_patterns(patterns)

        for doc in nlp.pipe(texts, n_process=2):
            for ent in doc.spans["ruler"]:
                assert ent.label_ == "FASTFOOD"


def test_span_ruler_serialize_dir(patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(patterns)
    with make_tempdir() as d:
        ruler.to_disk(d / "test_ruler")
        ruler.from_disk(d / "test_ruler")  # read from an existing directory
        with pytest.raises(ValueError):
            ruler.from_disk(d / "non_existing_dir")  # read from a bad directory


def test_span_ruler_remove_basic(person_org_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(person_org_patterns)
    doc = ruler(nlp.make_doc("Dina went to school"))
    assert len(ruler.patterns) == 3
    assert len(doc.spans["ruler"]) == 1
    assert doc.spans["ruler"][0].label_ == "PERSON"
    assert doc.spans["ruler"][0].text == "Dina"
    ruler.remove("PERSON")
    doc = ruler(nlp.make_doc("Dina went to school"))
    assert len(doc.spans["ruler"]) == 0
    assert len(ruler.patterns) == 2


def test_span_ruler_remove_nonexisting_pattern(person_org_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(person_org_patterns)
    assert len(ruler.patterns) == 3
    with pytest.raises(ValueError):
        ruler.remove("NE")
    with pytest.raises(ValueError):
        ruler.remove_by_id("NE")


def test_span_ruler_remove_several_patterns(person_org_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(person_org_patterns)
    doc = ruler(nlp.make_doc("Dina founded the company ACME."))
    assert len(ruler.patterns) == 3
    assert len(doc.spans["ruler"]) == 2
    assert doc.spans["ruler"][0].label_ == "PERSON"
    assert doc.spans["ruler"][0].text == "Dina"
    assert doc.spans["ruler"][1].label_ == "ORG"
    assert doc.spans["ruler"][1].text == "ACME"
    ruler.remove("PERSON")
    doc = ruler(nlp.make_doc("Dina founded the company ACME"))
    assert len(ruler.patterns) == 2
    assert len(doc.spans["ruler"]) == 1
    assert doc.spans["ruler"][0].label_ == "ORG"
    assert doc.spans["ruler"][0].text == "ACME"
    ruler.remove("ORG")
    with pytest.warns(UserWarning):
        doc = ruler(nlp.make_doc("Dina founded the company ACME"))
        assert len(ruler.patterns) == 0
        assert len(doc.spans["ruler"]) == 0


def test_span_ruler_remove_patterns_in_a_row(person_org_date_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(person_org_date_patterns)
    doc = ruler(nlp.make_doc("Dina founded the company ACME on June 14th"))
    assert len(doc.spans["ruler"]) == 3
    assert doc.spans["ruler"][0].label_ == "PERSON"
    assert doc.spans["ruler"][0].text == "Dina"
    assert doc.spans["ruler"][1].label_ == "ORG"
    assert doc.spans["ruler"][1].text == "ACME"
    assert doc.spans["ruler"][2].label_ == "DATE"
    assert doc.spans["ruler"][2].text == "June 14th"
    ruler.remove("ORG")
    ruler.remove("DATE")
    doc = ruler(nlp.make_doc("Dina went to school"))
    assert len(doc.spans["ruler"]) == 1


def test_span_ruler_remove_all_patterns(person_org_date_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(person_org_date_patterns)
    assert len(ruler.patterns) == 4
    ruler.remove("PERSON")
    assert len(ruler.patterns) == 3
    ruler.remove("ORG")
    assert len(ruler.patterns) == 1
    ruler.remove("DATE")
    assert len(ruler.patterns) == 0
    with pytest.warns(UserWarning):
        doc = ruler(nlp.make_doc("Dina founded the company ACME on June 14th"))
        assert len(doc.spans["ruler"]) == 0


def test_span_ruler_remove_and_add():
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    patterns1 = [{"label": "DATE1", "pattern": "last time"}]
    ruler.add_patterns(patterns1)
    doc = ruler(
        nlp.make_doc("I saw him last time we met, this time he brought some flowers")
    )
    assert len(ruler.patterns) == 1
    assert len(doc.spans["ruler"]) == 1
    assert doc.spans["ruler"][0].label_ == "DATE1"
    assert doc.spans["ruler"][0].text == "last time"
    patterns2 = [{"label": "DATE2", "pattern": "this time"}]
    ruler.add_patterns(patterns2)
    doc = ruler(
        nlp.make_doc("I saw him last time we met, this time he brought some flowers")
    )
    assert len(ruler.patterns) == 2
    assert len(doc.spans["ruler"]) == 2
    assert doc.spans["ruler"][0].label_ == "DATE1"
    assert doc.spans["ruler"][0].text == "last time"
    assert doc.spans["ruler"][1].label_ == "DATE2"
    assert doc.spans["ruler"][1].text == "this time"
    ruler.remove("DATE1")
    doc = ruler(
        nlp.make_doc("I saw him last time we met, this time he brought some flowers")
    )
    assert len(ruler.patterns) == 1
    assert len(doc.spans["ruler"]) == 1
    assert doc.spans["ruler"][0].label_ == "DATE2"
    assert doc.spans["ruler"][0].text == "this time"
    ruler.add_patterns(patterns1)
    doc = ruler(
        nlp.make_doc("I saw him last time we met, this time he brought some flowers")
    )
    assert len(ruler.patterns) == 2
    assert len(doc.spans["ruler"]) == 2
    patterns3 = [{"label": "DATE3", "pattern": "another time"}]
    ruler.add_patterns(patterns3)
    doc = ruler(
        nlp.make_doc(
            "I saw him last time we met, this time he brought some flowers, another time some chocolate."
        )
    )
    assert len(ruler.patterns) == 3
    assert len(doc.spans["ruler"]) == 3
    ruler.remove("DATE3")
    doc = ruler(
        nlp.make_doc(
            "I saw him last time we met, this time he brought some flowers, another time some chocolate."
        )
    )
    assert len(ruler.patterns) == 2
    assert len(doc.spans["ruler"]) == 2


def test_span_ruler_spans_filter(overlapping_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe(
        "span_ruler",
        config={"spans_filter": {"@misc": "spacy.first_longest_spans_filter.v1"}},
    )
    ruler.add_patterns(overlapping_patterns)
    doc = ruler(nlp.make_doc("foo bar baz"))
    assert len(doc.spans["ruler"]) == 1
    assert doc.spans["ruler"][0].label_ == "FOOBAR"


def test_span_ruler_ents_default_filter(overlapping_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler", config={"annotate_ents": True})
    ruler.add_patterns(overlapping_patterns)
    doc = ruler(nlp.make_doc("foo bar baz"))
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == "FOOBAR"


def test_span_ruler_ents_overwrite_filter(overlapping_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe(
        "span_ruler",
        config={
            "annotate_ents": True,
            "overwrite": False,
            "ents_filter": {"@misc": "spacy.prioritize_new_ents_filter.v1"},
        },
    )
    ruler.add_patterns(overlapping_patterns)
    # overlapping ents are clobbered, non-overlapping ents are preserved
    doc = nlp.make_doc("foo bar baz a b c")
    doc.ents = [Span(doc, 1, 3, label="BARBAZ"), Span(doc, 3, 6, label="ABC")]
    doc = ruler(doc)
    assert len(doc.ents) == 2
    assert doc.ents[0].label_ == "FOOBAR"
    assert doc.ents[1].label_ == "ABC"


def test_span_ruler_ents_bad_filter(overlapping_patterns):
    @registry.misc("test_pass_through_filter")
    def make_pass_through_filter():
        def pass_through_filter(spans1, spans2):
            return spans1 + spans2

        return pass_through_filter

    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe(
        "span_ruler",
        config={
            "annotate_ents": True,
            "ents_filter": {"@misc": "test_pass_through_filter"},
        },
    )
    ruler.add_patterns(overlapping_patterns)
    with pytest.raises(ValueError):
        ruler(nlp.make_doc("foo bar baz"))


@pytest.fixture
def exclude_org_patterns():
    return [
        {"label": "ORG", "pattern": "Apple"},
        {"label": "ORG", "pattern": "Microsoft"},
        {"label": "PERSON", "pattern": "Dina"},
    ]


def test_span_ruler_exclude_default_empty(exclude_org_patterns):
    """The exclusion set is empty by default and does not change results."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    assert ruler.exclude_patterns == []
    assert ruler.exclude_on == "overlap"
    ruler.add_patterns(exclude_org_patterns)
    doc = ruler(nlp.make_doc("Dina joined Apple and Microsoft"))
    assert [(s.text, s.label_) for s in doc.spans["ruler"]] == [
        ("Dina", "PERSON"),
        ("Apple", "ORG"),
        ("Microsoft", "ORG"),
    ]


def test_span_ruler_exclude_phrase_pattern(exclude_org_patterns):
    """A phrase exclusion filters the overlapping candidate but is never
    written to doc.spans itself."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    ruler.add_exclude_patterns([{"pattern": "fake Apple", "id": "fake"}])
    doc = ruler(nlp.make_doc("Dina joined fake Apple and Microsoft"))
    # "Dina" (PERSON) and "Microsoft" (ORG) survive; "Apple" is excluded;
    # the exclusion region "fake Apple" is not annotated.
    assert [(s.text, s.label_) for s in doc.spans["ruler"]] == [
        ("Dina", "PERSON"),
        ("Microsoft", "ORG"),
    ]
    assert all(s.text != "fake Apple" for s in doc.spans["ruler"])
    assert len(ruler.exclude_patterns) == 1


def test_span_ruler_exclude_token_pattern(exclude_org_patterns):
    """Token patterns can be used as exclusion patterns."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    # boundary-adjacent: "fake" ends right before "Apple" -> no exclusion
    ruler.add_exclude_patterns([{"pattern": [{"LOWER": "fake"}], "id": "adj"}])
    doc = ruler(nlp.make_doc("fake Apple"))
    assert [s.text for s in doc.spans["ruler"]] == ["Apple"]
    ruler.clear()
    ruler.add_patterns(exclude_org_patterns)
    # overlapping: "fake Apple" covers the "Apple" candidate -> excluded
    ruler.add_exclude_patterns(
        [{"pattern": [{"LOWER": "fake"}, {"LOWER": "apple"}], "id": "over"}]
    )
    doc = ruler(nlp.make_doc("fake Apple"))
    assert len(doc.spans["ruler"]) == 0


def test_span_ruler_exclude_boundary_adjacent(exclude_org_patterns):
    """Candidates that merely touch an exclusion region at a boundary are
    not excluded (overlap mode)."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    ruler.add_exclude_patterns([{"pattern": "Apple", "id": "apple"}])
    doc = ruler(nlp.make_doc("Microsoft Apple Dina"))
    # Microsoft is immediately before the exclusion region "Apple" and Dina
    # immediately after it; both are boundary-adjacent and survive.
    assert [(s.text, s.label_) for s in doc.spans["ruler"]] == [
        ("Microsoft", "ORG"),
        ("Dina", "PERSON"),
    ]


def test_span_ruler_exclude_partial_overlap():
    """In 'overlap' mode a partially overlapping candidate is excluded,
    while a non-overlapping candidate in the same doc survives."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(
        [
            {"label": "COMPANY", "pattern": "Apple Corp"},
            {"label": "ORG", "pattern": "Microsoft"},
        ]
    )
    # exclusion region "Corp" is nested inside / partially overlapping the
    # candidate "Apple Corp" (0,2)
    ruler.add_exclude_patterns([{"pattern": "Corp", "id": "corp"}])
    doc = ruler(nlp.make_doc("Apple Corp and Microsoft"))
    assert [(s.text, s.label_) for s in doc.spans["ruler"]] == [
        ("Microsoft", "ORG")
    ]
    # without the exclusion region in the doc, the candidate survives
    doc = ruler(nlp.make_doc("Apple Corp"))
    assert len(doc.spans["ruler"]) == 0
    ruler.remove_exclude_by_id("corp")
    doc = ruler(nlp.make_doc("Apple Corp"))
    assert [s.text for s in doc.spans["ruler"]] == ["Apple Corp"]


def test_span_ruler_exclude_contained_mode():
    """In 'contained' mode only candidates fully within an exclusion region
    are excluded; partially overlapping candidates survive."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe(
        "span_ruler", config={"exclude_on": "contained"}
    )
    assert ruler.exclude_on == "contained"
    ruler.add_patterns([{"label": "COMPANY", "pattern": "Apple Corp"}])
    # "Corp" (1,2) is nested in the candidate "Apple Corp" (0,2): the
    # candidate is not fully contained in the exclusion region, so it
    # survives; the exclusion region itself is never annotated.
    ruler.add_exclude_patterns([{"pattern": "Corp", "id": "corp"}])
    doc = ruler(nlp.make_doc("Apple Corp"))
    assert [(s.text, s.label_) for s in doc.spans["ruler"]] == [
        ("Apple Corp", "COMPANY")
    ]
    # a larger exclusion region fully contains the candidate -> excluded
    ruler2 = nlp.add_pipe(
        "span_ruler",
        name="span_ruler2",
        config={"exclude_on": "contained", "spans_key": "ruler2"},
    )
    ruler2.add_patterns([{"label": "ORG", "pattern": "Apple"}])
    ruler2.add_exclude_patterns([{"pattern": "fake Apple company", "id": "ctx"}])
    doc = ruler2(nlp.make_doc("the fake Apple company"))
    assert len(doc.spans["ruler2"]) == 0


def test_span_ruler_exclude_nested_regions(exclude_org_patterns):
    """Exclusion regions nested inside candidates: overlap mode removes the
    enclosing candidate, contained mode keeps it; a candidate nested in the
    exclusion region is removed in both modes."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns([{"label": "COMPANY", "pattern": "Apple Corp"}])
    ruler.add_exclude_patterns([{"pattern": "Corp", "id": "nested"}])
    doc = ruler(nlp.make_doc("Apple Corp"))
    assert len(doc.spans["ruler"]) == 0
    ruler.remove_exclude_by_id("nested")
    ruler.exclude_on = "contained"
    ruler.add_exclude_patterns([{"pattern": "Corp", "id": "nested"}])
    doc = ruler(nlp.make_doc("Apple Corp"))
    assert [s.text for s in doc.spans["ruler"]] == ["Apple Corp"]
    # candidate fully inside the exclusion region is excluded in both modes
    ruler.clear()
    ruler.add_patterns(exclude_org_patterns)
    ruler.add_exclude_patterns([{"pattern": "fake Apple now", "id": "outer"}])
    doc = ruler(nlp.make_doc("a fake Apple now b"))
    assert len(doc.spans["ruler"]) == 0


def test_span_ruler_exclude_labels(exclude_org_patterns):
    """An exclusion can be restricted to specific output labels."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    # region spanning both candidates, only targeting ORG
    ruler.add_exclude_patterns(
        [{"pattern": "Apple Dina", "labels": ["ORG"], "id": "org_only"}]
    )
    doc = ruler(nlp.make_doc("Apple Dina"))
    # Apple/ORG is excluded, Dina/PERSON is kept
    assert [(s.text, s.label_) for s in doc.spans["ruler"]] == [
        ("Dina", "PERSON")
    ]
    # same setup in contained mode: both are contained, but only ORG matches
    ruler2 = nlp.add_pipe(
        "span_ruler",
        name="ruler2",
        config={"exclude_on": "contained", "spans_key": "ruler2"},
    )
    ruler2.add_patterns(exclude_org_patterns)
    ruler2.add_exclude_patterns(
        [{"pattern": "Apple Dina", "labels": ["ORG"], "id": "org_only"}]
    )
    doc = ruler2(nlp.make_doc("Apple Dina"))
    assert [(s.text, s.label_) for s in doc.spans["ruler2"]] == [
        ("Dina", "PERSON")
    ]
    # an exclusion without "labels" applies to all labels
    ruler3 = nlp.add_pipe(
        "span_ruler", name="ruler3", config={"spans_key": "ruler3"}
    )
    ruler3.add_patterns(exclude_org_patterns)
    ruler3.add_exclude_patterns([{"pattern": "Apple Dina", "id": "all"}])
    doc = ruler3(nlp.make_doc("Apple Dina"))
    assert len(doc.spans["ruler3"]) == 0


def test_span_ruler_exclude_invalid_labels():
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    with pytest.raises(ValueError):
        ruler.add_exclude_patterns([{"pattern": "Apple", "labels": "ORG"}])
    with pytest.raises(ValueError):
        ruler.add_exclude_patterns([{"pattern": "Apple", "labels": [1]}])
    with pytest.raises(ValueError):
        ruler.add_exclude_patterns([{"labels": ["ORG"]}])
    with pytest.raises(ValueError):
        ruler.add_exclude_patterns([{"pattern": 42}])


def test_span_ruler_exclude_invalid_mode():
    nlp = spacy.blank("xx")
    with pytest.raises(ValueError):
        nlp.add_pipe("span_ruler", config={"exclude_on": "PARTIAL"})


def test_span_ruler_exclude_duplicate_hits_preserve_order(exclude_org_patterns):
    """Repeated hits of the same exclusion region are deduplicated and cannot
    change the order or multiplicity of the surviving candidates."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    # add the same region multiple times (phrase + token pattern)
    ruler.add_exclude_patterns(
        [
            {"pattern": "Apple", "id": "ex1"},
            {"pattern": "Apple", "id": "ex2"},
            {"pattern": [{"LOWER": "apple"}], "id": "ex3"},
        ]
    )
    doc = ruler(nlp.make_doc("Apple Microsoft Dina Apple Microsoft"))
    # both "Apple" hits are excluded; remaining candidates keep their sorted
    # order and are not duplicated
    assert [(s.text, s.label_) for s in doc.spans["ruler"]] == [
        ("Microsoft", "ORG"),
        ("Dina", "PERSON"),
        ("Microsoft", "ORG"),
    ]


def test_span_ruler_exclude_remove(exclude_org_patterns):
    """Exclusion patterns can be removed by ID and filtering stops."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    ruler.add_exclude_patterns(
        [
            {"pattern": "Apple", "id": "apple"},
            {"pattern": "Dina", "labels": ["PERSON"], "id": "dina"},
        ]
    )
    doc = ruler(nlp.make_doc("Apple Dina Microsoft"))
    assert [s.text for s in doc.spans["ruler"]] == ["Microsoft"]
    ruler.remove_exclude_by_id("apple")
    assert [p.get("id") for p in ruler.exclude_patterns] == ["dina"]
    doc = ruler(nlp.make_doc("Apple Dina Microsoft"))
    assert [s.text for s in doc.spans["ruler"]] == ["Apple", "Microsoft"]
    ruler.remove_exclude_by_id("dina")
    assert ruler.exclude_patterns == []
    doc = ruler(nlp.make_doc("Apple Dina Microsoft"))
    assert [s.text for s in doc.spans["ruler"]] == ["Apple", "Dina", "Microsoft"]
    with pytest.raises(ValueError):
        ruler.remove_exclude_by_id("missing")


def test_span_ruler_exclude_clear(exclude_org_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    ruler.add_exclude_patterns([{"pattern": "Apple", "id": "apple"}])
    assert len(ruler.exclude_patterns) == 1
    ruler.clear()
    assert ruler.patterns == []
    assert ruler.exclude_patterns == []


def test_span_ruler_exclude_initialize_clears(exclude_org_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    ruler.add_exclude_patterns([{"pattern": "Apple", "id": "apple"}])
    ruler.initialize(lambda: [], patterns=[{"label": "ORG", "pattern": "Apple"}])
    assert len(ruler.patterns) == 1
    assert ruler.exclude_patterns == []


def test_span_ruler_exclude_ents(exclude_org_patterns):
    """Exclusions also filter candidates written to doc.ents, never add
    entities themselves, and leave pre-existing ents alone."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe(
        "span_ruler",
        config={
            "annotate_ents": True,
            "overwrite": False,
            "spans_key": None,
        },
    )
    ruler.add_patterns(exclude_org_patterns)
    ruler.add_exclude_patterns([{"pattern": "Apple", "id": "apple"}])
    doc = nlp.make_doc("Apple Microsoft Dina")
    # pre-existing entity inside the exclusion region is preserved: exclusion
    # only filters the ruler's own candidate matches.
    doc.ents = [Span(doc, 1, 2, label="EXISTING")]
    doc = ruler(doc)
    assert [(e.text, e.label_) for e in doc.ents] == [
        ("Microsoft", "EXISTING"),
        ("Dina", "PERSON"),
    ]
    assert "Apple" not in [e.text for e in doc.ents]
    assert len(doc.spans) == 0


def test_span_ruler_exclude_with_spans_filter(overlapping_patterns):
    """The surviving candidates still go through the original spans_filter."""
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe(
        "span_ruler",
        config={"spans_filter": {"@misc": "spacy.first_longest_spans_filter.v1"}},
    )
    ruler.add_patterns(overlapping_patterns)
    # "foo" only overlaps FOOBAR and is boundary-adjacent to BARBAZ, so
    # BARBAZ survives and is still processed by the spans_filter
    ruler.add_exclude_patterns([{"pattern": "foo", "id": "foo"}])
    doc = ruler(nlp.make_doc("foo bar baz"))
    # FOOBAR is excluded; BARBAZ survives and is the only span written
    assert [(s.text, s.label_) for s in doc.spans["ruler"]] == [
        ("bar baz", "BARBAZ")
    ]


def test_span_ruler_exclude_serialize_bytes(exclude_org_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler", config={"exclude_on": "contained"})
    ruler.add_patterns(exclude_org_patterns)
    exclude_patterns = [
        {"pattern": "Apple", "labels": ["ORG"], "id": "apple"},
        {"pattern": [{"LOWER": "dina"}], "id": "dina"},
    ]
    ruler.add_exclude_patterns(exclude_patterns)
    ruler_bytes = ruler.to_bytes()

    # exclude_on is a config setting (not part of the serialized patterns),
    # so the receiving pipe is created with the same config
    new_ruler = nlp.add_pipe(
        "span_ruler",
        name="new_ruler",
        config={"exclude_on": "contained", "spans_key": "new_ruler"},
    )
    new_ruler.from_bytes(ruler_bytes)
    assert new_ruler.exclude_on == "contained"
    assert new_ruler.patterns == ruler.patterns
    assert new_ruler.exclude_patterns == exclude_patterns
    doc = new_ruler(nlp.make_doc("Apple Dina Microsoft"))
    assert [(s.text, s.label_) for s in doc.spans["new_ruler"]] == [
        ("Microsoft", "ORG")
    ]
    # "Apple Corp" only partially overlaps the "Corp" exclusion -> kept in
    # contained mode
    new_ruler.clear()
    new_ruler.add_patterns([{"label": "COMPANY", "pattern": "Apple Corp"}])
    new_ruler.add_exclude_patterns([{"pattern": "Corp", "id": "corp"}])
    data = new_ruler.to_bytes()
    ruler3 = nlp.add_pipe(
        "span_ruler",
        name="ruler3",
        config={"exclude_on": "contained", "spans_key": "ruler3"},
    )
    ruler3.from_bytes(data)
    doc = ruler3(nlp.make_doc("Apple Corp"))
    assert [s.text for s in doc.spans["ruler3"]] == ["Apple Corp"]


def test_span_ruler_exclude_serialize_bytes_exclude_kwarg(exclude_org_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    ruler.add_exclude_patterns([{"pattern": "Apple", "id": "apple"}])
    ruler_bytes = ruler.to_bytes(exclude=["exclude_patterns"])
    new_ruler = nlp.add_pipe("span_ruler", name="no_excludes")
    new_ruler.from_bytes(ruler_bytes)
    assert len(new_ruler.patterns) == len(exclude_org_patterns)
    assert new_ruler.exclude_patterns == []


def test_span_ruler_exclude_serialize_dir(exclude_org_patterns):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    exclude_patterns = [
        {"pattern": "Apple", "labels": ["ORG"], "id": "apple"},
    ]
    ruler.add_exclude_patterns(exclude_patterns)
    with make_tempdir() as d:
        ruler.to_disk(d / "test_ruler")
        assert (d / "test_ruler" / "exclude_patterns").exists()
        new_ruler = nlp.add_pipe(
            "span_ruler", name="disk_ruler", config={"spans_key": "disk_ruler"}
        )
        new_ruler.from_disk(d / "test_ruler")
        assert new_ruler.patterns == ruler.patterns
        assert new_ruler.exclude_patterns == exclude_patterns
        doc = new_ruler(nlp.make_doc("Apple Dina"))
        assert [(s.text, s.label_) for s in doc.spans["disk_ruler"]] == [
            ("Dina", "PERSON")
        ]


def test_span_ruler_exclude_load_old_model_bytes(exclude_org_patterns):
    """Bytes produced without an 'exclude_patterns' block (old models) load
    with an empty exclusion set."""
    import srsly

    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(exclude_org_patterns)
    old_bytes = srsly.msgpack_dumps(
        {"patterns": srsly.json_dumps(exclude_org_patterns)}
    )
    ruler.add_exclude_patterns([{"pattern": "Apple", "id": "apple"}])
    ruler.from_bytes(old_bytes)
    assert ruler.exclude_patterns == []
    assert len(ruler.patterns) == len(exclude_org_patterns)
    doc = ruler(nlp.make_doc("Apple Dina"))
    assert [s.text for s in doc.spans["ruler"]] == ["Apple", "Dina"]


def test_span_ruler_exclude_load_old_model_dir(exclude_org_patterns):
    """Directories without an 'exclude_patterns' file (old models) load with
    an empty exclusion set."""
    import srsly

    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("span_ruler")
    with make_tempdir() as d:
        model_dir = d / "old_ruler"
        model_dir.mkdir()
        srsly.write_jsonl(model_dir / "patterns", exclude_org_patterns)
        assert not (model_dir / "exclude_patterns").exists()
        ruler.from_disk(model_dir)
    assert ruler.exclude_patterns == []
    assert [p["label"] for p in ruler.patterns] == [
        p["label"] for p in exclude_org_patterns
    ]
    doc = ruler(nlp.make_doc("Apple Dina Microsoft"))
    assert [s.text for s in doc.spans["ruler"]] == ["Apple", "Dina", "Microsoft"]
