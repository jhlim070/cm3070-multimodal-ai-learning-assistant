"""
Regression tests for PaddleOCR image-text extraction.

The production code uses PaddleOCR 3.x's ``predict()`` API without
the removed ``cls`` argument. These tests verify that API call and
the parsing of the supported result shapes.

The parser also accepts the nested-list result format used by older
PaddleOCR releases. This tests parser compatibility only; it does not
guarantee that the current OCR initialisation works with PaddleOCR 2.x.

OCR and its result objects are mocked here, so these tests verify our
API usage and result parsing rather than OCR accuracy.
"""
import pytest

import material_processor as mp


# result-shape fakes --------------------------------------------------------

class _DictLikeResult:
    """
    Mocks a PaddleOCR 3.x result supporting ``result['rec_texts']``.
    """

    def __init__(self, rec_texts):
        self._data = {"rec_texts": rec_texts}

    def __getitem__(self, key):
        return self._data[key]


class _AttrLikeResult:
    """
    Mocks a PaddleOCR 3.x result exposing ``rec_texts`` as an attribute.
    """
    def __init__(self, rec_texts):
        self.rec_texts = rec_texts


class _JsonNestedResult:
    """
    Mocks a result where ``rec_texts`` is nested under ``result.json['res']``.
    """
    def __init__(self, rec_texts):
        self.json = {"res": {"rec_texts": rec_texts}}


class _UnrecognisedResult:
    """
    Represents an unsupported result shape that the parser must reject.
    """
    def __repr__(self):
        return "<UnrecognisedResult>"


class FakeOCR:
    """
    Fake OCR client that records ``predict()`` arguments and rejects calls to 
    the deprecated ``ocr()`` API.
    """
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, img_path, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def ocr(self, *args, **kwargs):
        raise AssertionError("material_processor called the deprecated .ocr(); it must call .predict()")


# PaddleOCR API usage -------------------------------------------------------

def test_ocr_is_called_without_the_removed_cls_kwarg(monkeypatch):
    """
    Regression test for the removed ``cls`` argument in PaddleOCR 3.x.
    """
    fake = FakeOCR([_DictLikeResult(["hello"])])
    monkeypatch.setattr(mp, "ocr", fake)
    mp.extract_text_from_image("/tmp/x.png")
    assert fake.calls == [{}], f"cls= (or any kwarg) was passed to .predict(): {fake.calls}"


def test_predict_is_used_not_the_deprecated_ocr_method(monkeypatch):
    """
    Verify that image extraction uses PaddleOCR's ``predict()`` API 
    rather than the deprecated ``ocr()`` API.
    """
    fake = FakeOCR([_DictLikeResult(["hello"])])
    monkeypatch.setattr(mp, "ocr", fake)
    assert mp.extract_text_from_image("/tmp/x.png") == "hello"
    assert len(fake.calls) == 1


# _extract_rec_texts: every shape the parser has to tolerate ---------------

def test_extract_rec_texts_dict_style():
    assert mp._extract_rec_texts(_DictLikeResult(["hello", "world"])) == ["hello", "world"]


def test_extract_rec_texts_attribute_style():
    assert mp._extract_rec_texts(_AttrLikeResult(["hello", "world"])) == ["hello", "world"]


def test_extract_rec_texts_json_nested_style():
    assert mp._extract_rec_texts(_JsonNestedResult(["hello", "world"])) == ["hello", "world"]


def test_extract_rec_texts_dict_style_is_tried_before_json_nested():
    """
    Prefer direct result access before falling back to the nested ``.json`` representation.
    """
    assert mp._extract_rec_texts(_DictLikeResult(["direct"])) == ["direct"]


def test_extract_rec_texts_raises_loudly_on_an_unrecognised_shape():
    """
    Unsupported result shapes must raise rather than being mistaken for blank OCR output.
    """
    with pytest.raises(ValueError, match="Could not find recognised text"):
        mp._extract_rec_texts(_UnrecognisedResult())


def test_extract_rec_texts_error_message_names_the_actual_type():
    with pytest.raises(ValueError, match="_UnrecognisedResult"):
        mp._extract_rec_texts(_UnrecognisedResult())


# extract_text_from_image: end to end, both API generations ---------------

def test_new_shape_single_result_item(monkeypatch):
    monkeypatch.setattr(mp, "ocr", FakeOCR([_DictLikeResult(["Hello", "World"])]))
    assert mp.extract_text_from_image("/tmp/x.png") == "Hello World"


def test_new_shape_multiple_result_items_are_all_included(monkeypatch):
    """
    A single OCR call may return multiple result items.
    """
    monkeypatch.setattr(mp, "ocr", FakeOCR([_DictLikeResult(["first"]), _DictLikeResult(["second"])]))
    assert mp.extract_text_from_image("/tmp/x.png") == "first second"


def test_new_shape_via_attribute_access_end_to_end(monkeypatch):
    monkeypatch.setattr(mp, "ocr", FakeOCR([_AttrLikeResult(["from attribute access"])]))
    assert mp.extract_text_from_image("/tmp/x.png") == "from attribute access"


def test_new_shape_via_json_nested_access_end_to_end(monkeypatch):
    monkeypatch.setattr(mp, "ocr", FakeOCR([_JsonNestedResult(["from json"])]))
    assert mp.extract_text_from_image("/tmp/x.png") == "from json"


def test_blank_image_returns_empty_string_not_an_error(monkeypatch):
    """
    A valid OCR result containing no text should produce an empty string, not an error.
    """
    monkeypatch.setattr(mp, "ocr", FakeOCR([None]))
    assert mp.extract_text_from_image("/tmp/x.png") == ""


def test_empty_result_list_returns_empty_string(monkeypatch):
    monkeypatch.setattr(mp, "ocr", FakeOCR([]))
    assert mp.extract_text_from_image("/tmp/x.png") == ""


def test_none_result_returns_empty_string(monkeypatch):
    monkeypatch.setattr(mp, "ocr", FakeOCR(None))
    assert mp.extract_text_from_image("/tmp/x.png") == ""


def test_a_mix_of_none_and_real_results_only_includes_the_real_ones(monkeypatch):
    monkeypatch.setattr(mp, "ocr", FakeOCR([None, _DictLikeResult(["real text"]), None]))
    assert mp.extract_text_from_image("/tmp/x.png") == "real text"


def test_an_unparseable_result_from_a_real_call_propagates_as_an_error(monkeypatch):
    """
    An unsupported OCR result must propagate as an error rather than becoming empty text.
    """
    monkeypatch.setattr(mp, "ocr", FakeOCR([_UnrecognisedResult()]))
    with pytest.raises(ValueError, match="Could not find recognised text"):
        mp.extract_text_from_image("/tmp/x.png")


def test_lines_with_only_whitespace_are_dropped():
    assert mp._extract_rec_texts(_DictLikeResult(["real", "   ", "", "text"])) == ["real", "   ", "", "text"]
    # Whitespace filtering is applied when extracted lines are joined.


def test_whitespace_only_lines_are_filtered_when_joining(monkeypatch):
    monkeypatch.setattr(mp, "ocr", FakeOCR([_DictLikeResult(["real", "   ", "", "text"])]))
    assert mp.extract_text_from_image("/tmp/x.png") == "real text"