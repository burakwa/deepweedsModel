"""config.py testleri: sınıf kataloğu, jsonable, save_json, build_label_map."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import CLASS_NAMES, NUM_CLASSES, build_label_map, jsonable, save_json


def test_num_classes_matches_catalog():
    assert NUM_CLASSES == len(CLASS_NAMES) == 9
    assert "Negative" in CLASS_NAMES


def test_jsonable_dataclass_and_paths(tmp_path):
    @dataclass
    class Cfg:
        path: Path = Path("a/b")
        n: int = 3
        tag: str = "merhaba"

    out = jsonable(Cfg())
    assert out == {"path": "a" + str(Path("/b")), "n": 3, "tag": "merhaba"}


def test_jsonable_nested_containers():
    obj = {"a": [Path("x"), (np.float32(1.5), True)], "b": None}
    out = jsonable(obj)
    assert out["a"][0] == "x"
    assert out["a"][1][0] == 1.5 and isinstance(out["a"][1][0], float)
    assert out["a"][1][1] is True
    assert out["b"] is None


def test_jsonable_numpy_scalar():
    assert jsonable(np.float64(2.5)) == 2.5
    assert jsonable(np.int32(7)) == 7


def test_save_json_creates_parents_and_roundtrip(tmp_path):
    target = tmp_path / "a" / "b" / "out.json"
    save_json(target, {"ad": "Türkçe ĞÜŞİÖÇ", "p": Path("runs/x")})
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["ad"] == "Türkçe ĞÜŞİÖÇ"
    assert data["p"] == "runs" + str(Path("/x"))


def test_build_label_map_numeric_identity():
    series = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert build_label_map(series) == {i: i for i in range(9)}


def test_build_label_map_string_names():
    series = pd.Series(list(CLASS_NAMES))
    m = build_label_map(series)
    for i, name in enumerate(CLASS_NAMES):
        assert m[name] == i


def test_build_label_map_unknown_label_raises():
    series = pd.Series(["Bilinmeyen Otu", *CLASS_NAMES[1:]])
    with pytest.raises(ValueError):
        build_label_map(series)
