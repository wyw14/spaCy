import zlib

import numpy as np
import pytest
import srsly

import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.tokens.underscore import Underscore


@pytest.mark.issue(4367)
def test_issue4367():
    """Test that docbin init goes well"""
    DocBin()
    DocBin(attrs=["LEMMA"])
    DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"])


@pytest.mark.issue(4528)
def test_issue4528(en_vocab):
    """Test that user_data is correctly serialized in DocBin."""
    doc = Doc(en_vocab, words=["hello", "world"])
    doc.user_data["foo"] = "bar"
    # This is how extension attribute values are stored in the user data
    doc.user_data[("._.", "foo", None, None)] = "bar"
    doc_bin = DocBin(store_user_data=True)
    doc_bin.add(doc)
    doc_bin_bytes = doc_bin.to_bytes()
    new_doc_bin = DocBin(store_user_data=True).from_bytes(doc_bin_bytes)
    new_doc = list(new_doc_bin.get_docs(en_vocab))[0]
    assert new_doc.user_data["foo"] == "bar"
    assert new_doc.user_data[("._.", "foo", None, None)] == "bar"


@pytest.mark.issue(5141)
def test_issue5141(en_vocab):
    """Ensure an empty DocBin does not crash on serialization"""
    doc_bin = DocBin(attrs=["DEP", "HEAD"])
    assert list(doc_bin.get_docs(en_vocab)) == []
    doc_bin_bytes = doc_bin.to_bytes()
    doc_bin_2 = DocBin().from_bytes(doc_bin_bytes)
    assert list(doc_bin_2.get_docs(en_vocab)) == []


def test_serialize_doc_bin():
    doc_bin = DocBin(
        attrs=["LEMMA", "ENT_IOB", "ENT_TYPE", "NORM", "ENT_ID"], store_user_data=True
    )
    texts = ["Some text", "Lots of texts...", "..."]
    cats = {"A": 0.5}
    nlp = English()
    for doc in nlp.pipe(texts):
        doc.cats = cats
        span = doc[0:2]
        span.label_ = "UNUSUAL_SPAN_LABEL"
        span.id_ = "UNUSUAL_SPAN_ID"
        span.kb_id_ = "UNUSUAL_SPAN_KB_ID"
        doc.spans["start"] = [span]
        doc[0].norm_ = "UNUSUAL_TOKEN_NORM"
        doc[0].ent_id_ = "UNUSUAL_TOKEN_ENT_ID"
        doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()

    # Deserialize later, e.g. in a new process
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_bytes(bytes_data)
    reloaded_docs = list(doc_bin.get_docs(nlp.vocab))
    for i, doc in enumerate(reloaded_docs):
        assert doc.text == texts[i]
        assert doc.cats == cats
        assert len(doc.spans) == 1
        assert doc.spans["start"][0].label_ == "UNUSUAL_SPAN_LABEL"
        assert doc.spans["start"][0].id_ == "UNUSUAL_SPAN_ID"
        assert doc.spans["start"][0].kb_id_ == "UNUSUAL_SPAN_KB_ID"
        assert doc[0].norm_ == "UNUSUAL_TOKEN_NORM"
        assert doc[0].ent_id_ == "UNUSUAL_TOKEN_ENT_ID"


def test_serialize_doc_bin_unknown_spaces(en_vocab):
    doc1 = Doc(en_vocab, words=["that", "'s"])
    assert doc1.has_unknown_spaces
    assert doc1.text == "that 's "
    doc2 = Doc(en_vocab, words=["that", "'s"], spaces=[False, False])
    assert not doc2.has_unknown_spaces
    assert doc2.text == "that's"

    doc_bin = DocBin().from_bytes(DocBin(docs=[doc1, doc2]).to_bytes())
    re_doc1, re_doc2 = doc_bin.get_docs(en_vocab)
    assert re_doc1.has_unknown_spaces
    assert re_doc1.text == "that 's "
    assert not re_doc2.has_unknown_spaces
    assert re_doc2.text == "that's"


@pytest.mark.parametrize(
    "writer_flag,reader_flag,reader_value",
    [
        (True, True, "bar"),
        (True, False, "bar"),
        (False, True, "nothing"),
        (False, False, "nothing"),
    ],
)
def test_serialize_custom_extension(en_vocab, writer_flag, reader_flag, reader_value):
    """Test that custom extensions are correctly serialized in DocBin."""
    Doc.set_extension("foo", default="nothing")
    doc = Doc(en_vocab, words=["hello", "world"])
    doc._.foo = "bar"
    doc_bin_1 = DocBin(store_user_data=writer_flag)
    doc_bin_1.add(doc)
    doc_bin_bytes = doc_bin_1.to_bytes()
    doc_bin_2 = DocBin(store_user_data=reader_flag).from_bytes(doc_bin_bytes)
    doc_2 = list(doc_bin_2.get_docs(en_vocab))[0]
    assert doc_2._.foo == reader_value
    Underscore.doc_extensions = {}


def test_docbin_select_reorder_duplicates(en_vocab):
    """select returns documents in the requested order and supports
    repeated indices and generator inputs."""
    doc1 = Doc(en_vocab, words=["one", "two"], spaces=[True, False])
    doc2 = Doc(en_vocab, words=["three"], spaces=[False])
    doc3 = Doc(en_vocab, words=["four", "five", "six"], spaces=[True, True, False])
    doc_bin = DocBin(docs=[doc1, doc2, doc3])

    sub = doc_bin.select(i for i in [2, 0, 2])
    assert isinstance(sub, DocBin)
    assert sub is not doc_bin
    assert len(sub) == 3
    assert sub.attrs == doc_bin.attrs
    assert sub.store_user_data == doc_bin.store_user_data
    texts = [doc.text for doc in sub.get_docs(en_vocab)]
    assert texts == ["four five six", "one two", "four five six"]
    # The repeated document is included as two independent stored copies.
    assert sub.tokens[0] is not sub.tokens[2]


def test_docbin_select_empty(en_vocab):
    """An empty index sequence returns an empty DocBin that round-trips."""
    doc_bin = DocBin(docs=[Doc(en_vocab, words=["a"])])
    sub = doc_bin.select([])
    assert len(sub) == 0
    assert list(sub.get_docs(en_vocab)) == []
    assert sub.attrs == doc_bin.attrs
    assert sub.store_user_data == doc_bin.store_user_data
    reloaded = DocBin().from_bytes(sub.to_bytes())
    assert len(reloaded) == 0
    assert list(reloaded.get_docs(en_vocab)) == []

    # Selecting from an already-empty DocBin only works with an empty sequence.
    empty_bin = DocBin(attrs=["DEP", "HEAD"])
    assert len(empty_bin.select(())) == 0


def test_docbin_select_empty_doc(en_vocab):
    """Zero-length documents survive selection and reloading."""
    doc1 = Doc(en_vocab, words=["hello"], spaces=[False])
    doc2 = Doc(en_vocab, words=[])
    doc_bin = DocBin(docs=[doc1, doc2])

    sub = doc_bin.select([1, 0, 1])
    assert len(sub) == 3
    reloaded = DocBin().from_bytes(sub.to_bytes())
    docs = list(reloaded.get_docs(en_vocab))
    assert [len(doc) for doc in docs] == [0, 1, 0]
    assert [doc.text for doc in docs] == ["", "hello", ""]


@pytest.mark.parametrize("bad_indices", [[-1], [2], [0, -1], [1, 2], [2, 0]])
def test_docbin_select_invalid_indices(en_vocab, bad_indices):
    """Negative or out-of-range indices raise IndexError, and all indices
    are validated before anything is copied."""
    doc_bin = DocBin(
        docs=[Doc(en_vocab, words=["a"]), Doc(en_vocab, words=["b"])]
    )
    original_tokens = list(doc_bin.tokens)
    with pytest.raises(IndexError):
        doc_bin.select(bad_indices)
    # The source DocBin is untouched by the failed selection.
    assert len(doc_bin) == 2
    assert [id(array) for array in doc_bin.tokens] == [
        id(array) for array in original_tokens
    ]


@pytest.mark.parametrize("bad_value", [1.0, "0", None, [0], object()])
def test_docbin_select_non_integer_indices(en_vocab, bad_value):
    """Non-integer indices raise a TypeError."""
    doc_bin = DocBin(docs=[Doc(en_vocab, words=["a"])])
    with pytest.raises(TypeError):
        doc_bin.select([0, bad_value])
    assert len(doc_bin) == 1


def test_docbin_select_spans_user_data_roundtrip(en_vocab):
    """The selection preserves cats, SpanGroups, flags and user_data across
    serialization, and keeps the strings needed for get_docs."""
    doc1 = Doc(en_vocab, words=["hello", "world"], spaces=[True, False])
    doc1.cats = {"POSITIVE": 0.75}
    doc1.user_data["doc_key"] = {"a": 1}
    span = doc1[0:2]
    span.label_ = "UNUSUAL_SPAN_LABEL"
    span.kb_id_ = "UNUSUAL_SPAN_KB_ID"
    span.id_ = "UNUSUAL_SPAN_ID"
    doc1.spans["start"] = [span]

    doc2 = Doc(
        en_vocab, words=["second", "doc", "here"], spaces=[True, True, False]
    )
    doc2.cats = {"NEGATIVE": 0.9}
    doc2.user_data["doc_key"] = "second"
    doc2.spans["middle"] = [doc2[1:3]]

    doc_bin = DocBin(
        attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"],
        store_user_data=True,
        docs=[doc1, doc2],
    )
    sub = doc_bin.select([1, 0])
    assert sub.attrs == doc_bin.attrs
    assert sub.store_user_data is True

    reloaded = DocBin(store_user_data=True).from_bytes(sub.to_bytes())
    docs = list(reloaded.get_docs(en_vocab))
    assert docs[0].text == "second doc here"
    assert docs[0].cats == {"NEGATIVE": 0.9}
    assert docs[0].user_data["doc_key"] == "second"
    assert docs[0].spans["middle"][0].text == "doc here"
    assert docs[1].text == "hello world"
    assert docs[1].cats == {"POSITIVE": 0.75}
    assert docs[1].user_data["doc_key"] == {"a": 1}
    selected_span = docs[1].spans["start"][0]
    assert selected_span.label_ == "UNUSUAL_SPAN_LABEL"
    assert selected_span.kb_id_ == "UNUSUAL_SPAN_KB_ID"
    assert selected_span.id_ == "UNUSUAL_SPAN_ID"


def test_docbin_select_unknown_spaces_flags(en_vocab):
    """The has_unknown_spaces flags follow the selected documents."""
    doc1 = Doc(en_vocab, words=["that", "'s"])
    assert doc1.has_unknown_spaces
    doc2 = Doc(en_vocab, words=["that", "'s"], spaces=[False, False])
    assert not doc2.has_unknown_spaces
    doc_bin = DocBin().from_bytes(DocBin(docs=[doc1, doc2]).to_bytes())

    sub = doc_bin.select([1, 0])
    reloaded = DocBin().from_bytes(sub.to_bytes())
    first, second = reloaded.get_docs(en_vocab)
    assert not first.has_unknown_spaces
    assert first.text == "that's"
    assert second.has_unknown_spaces
    assert second.text == "that 's "


def test_docbin_select_store_user_data_setting(en_vocab):
    """The serialization of user_data keeps following the original
    store_user_data setting."""
    doc = Doc(en_vocab, words=["hello", "world"])
    doc.user_data["foo"] = "bar"

    stored = DocBin(store_user_data=True, docs=[doc])
    stored_sub = stored.select([0])
    assert stored_sub.store_user_data is True
    msg = srsly.msgpack_loads(zlib.decompress(stored_sub.to_bytes()))
    assert "user_data" in msg

    not_stored = DocBin(store_user_data=False, docs=[doc])
    not_stored_sub = not_stored.select([0])
    assert not_stored_sub.store_user_data is False
    msg = srsly.msgpack_loads(zlib.decompress(not_stored_sub.to_bytes()))
    assert "user_data" not in msg


def test_docbin_select_isolation(en_vocab):
    """Appending to or mutating one DocBin does not change the other, and
    repeated indices do not share mutable data."""
    doc1 = Doc(en_vocab, words=["one"], spaces=[False])
    doc1.cats = {"k": 0.1}
    doc2 = Doc(en_vocab, words=["two"], spaces=[False])
    doc2.cats = {"k": 0.2}
    doc3 = Doc(en_vocab, words=["three"], spaces=[False])
    doc3.cats = {"k": 0.3}
    doc_bin = DocBin(docs=[doc1, doc2, doc3])

    sub = doc_bin.select([0, 0])
    assert len(sub) == 2
    assert sub.cats[0] is not sub.cats[1]

    # Mutating and appending to the source does not affect the selection.
    sub_tokens = [array.copy() for array in sub.tokens]
    doc_bin.cats[0]["k"] = 9.9
    # A non-ORTH column, so get_docs keeps working after the mutation.
    doc_bin.tokens[0][0, -1] = 999
    doc_bin.add(Doc(en_vocab, words=["four"], spaces=[False]))
    assert len(doc_bin) == 4
    assert len(sub) == 2
    assert all(
        np.array_equal(copied, sub_tokens[i])
        for i, copied in enumerate(sub.tokens)
    )
    sub_docs = list(sub.get_docs(en_vocab))
    assert sub_docs[0].cats == {"k": 0.1}
    assert sub_docs[1].cats == {"k": 0.1}

    # Mutating and appending to the selection does not affect the source.
    sub.cats[0]["k"] = 7.7
    sub.tokens[0][0, -1] = 888
    sub.add(Doc(en_vocab, words=["five"], spaces=[False]))
    assert len(sub) == 3
    assert len(doc_bin) == 4
    assert doc_bin.tokens[0][0, -1] == 999
    source_docs = list(doc_bin.get_docs(en_vocab))
    assert source_docs[0].cats == {"k": 9.9}
    assert source_docs[1].cats == {"k": 0.2}
    assert source_docs[3].text == "four"
    remaining_sub_docs = list(sub.get_docs(en_vocab))
    assert remaining_sub_docs[1].cats == {"k": 0.1}
    assert remaining_sub_docs[2].text == "five"
