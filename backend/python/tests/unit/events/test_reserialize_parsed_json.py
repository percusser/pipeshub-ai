import pytest

from app.events.events import _reserialize_parsed_json


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (42, b"42"),
        (1.5, b"1.5"),
        (True, b"true"),
        ({"b": 1, "a": 2}, b'{"a": 2, "b": 1}'),
        ([1, "x"], b'[1, "x"]'),
    ],
)
def test_a_parsed_json_value_becomes_bytes(value: object, expected: bytes) -> None:
    assert _reserialize_parsed_json(value) == expected


@pytest.mark.parametrize("value", [None, b"raw", bytearray(b"raw"), "text"])
def test_raw_content_is_left_alone(value: object) -> None:
    assert _reserialize_parsed_json(value) is value
