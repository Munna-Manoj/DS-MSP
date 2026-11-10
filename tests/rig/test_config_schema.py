"""The calib_param.yml JSON Schema matches what the loader accepts and does."""
import dataclasses
import json
import warnings

import cv2
import pytest

from ds_msp.rig import calib_param, config_schema as cs

pytestmark = pytest.mark.req("FR-UI-002")

_TYPES = {"integer": int, "number": (int, float), "string": str, "boolean": bool,
          "array": list, "null": type(None)}


def _accepts(prop, value):
    types = prop["type"] if isinstance(prop["type"], list) else [prop["type"]]
    if isinstance(value, bool) and "boolean" not in types:
        return False
    return any(isinstance(value, _TYPES[t]) for t in types)


def test_schema_lists_exactly_the_keys_the_loader_knows():
    assert set(cs.PROPERTIES) == set(calib_param._KNOWN_KEYS)
    groups = {g["id"] for g in cs.GROUPS}
    for k, p in cs.PROPERTIES.items():
        assert p["x-group"] in groups, k
        assert p["x-level"] in ("basic", "expert"), k
        assert p["description"].endswith("."), k


def test_defaults_are_what_the_loader_uses_when_a_key_is_absent(tmp_path):
    path = tmp_path / "calib_param.yml"
    path.write_text("%YAML:1.0\nnumber_camera: 2\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = calib_param.load_config(str(path))
    fields = {f.name for f in dataclasses.fields(cfg)}
    checked = 0
    for key, prop in cs.PROPERTIES.items():
        if key in fields and key not in ("number_camera", "camera_models"):
            assert getattr(cfg, key) == prop["default"], key
            checked += 1
    board = cfg.boards[0]
    assert (board.n_x, board.n_y) == (cs.PROPERTIES["number_x_square"]["default"],
                                      cs.PROPERTIES["number_y_square"]["default"])
    assert board.length_square == cs.PROPERTIES["length_square"]["default"]
    assert board.length_marker == cs.PROPERTIES["length_marker"]["default"]
    assert board.square_size == cs.PROPERTIES["square_size"]["default"]
    assert cfg.camera_models == ["radtan", "radtan"]          # distortion_model 0 by default
    assert checked >= 35


@pytest.mark.parametrize("gate", ["audit_gate", "uncertainty_gate", "convergence_gate",
                                  "coverage_gate", "lens_gate", "crossval_gate"])
def test_gate_values_match_the_loader(gate, tmp_path):
    assert cs.PROPERTIES[gate]["enum"] == cs.GATE_VALUES
    path = tmp_path / "c.yml"
    path.write_text(f'%YAML:1.0\nnumber_camera: 1\n{gate}: "sometimes"\n')
    with pytest.raises(ValueError):
        calib_param.load_config(str(path))


def test_shipped_template_values_fit_the_schema():
    """Every value in the template --init-config writes has the schema's type."""
    import importlib.resources
    tpl = importlib.resources.files("ds_msp.rig") / "configs" / "calib_param.template.yml"
    with importlib.resources.as_file(tpl) as p:
        fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
        root = fs.root()
        seen = 0
        for key in root.keys():
            if key not in cs.PROPERTIES:
                continue
            node = root.getNode(key)
            if node.isSeq():
                value = [node.at(i).real() for i in range(node.size())]
            elif node.isString():
                s = node.string()
                value = {"true": True, "false": False, "None": None}.get(s, s)
            else:
                value = node.real()
                if float(value).is_integer() and "integer" in str(cs.PROPERTIES[key]["type"]):
                    value = int(value)
            assert _accepts(cs.PROPERTIES[key], value), (key, value)
            seen += 1
        fs.release()
    assert seen >= 20


def test_schema_is_plain_json():
    doc = cs.config_schema()
    assert json.loads(json.dumps(doc)) == doc
    assert doc["required"] == ["number_camera"]
    assert doc["x-schema-version"] == cs.SCHEMA_VERSION


@pytest.mark.req("FR-UI-014")
def test_schema_states_what_the_production_profile_changes():
    """A form shows what an untouched key does under ``profile: production``; the schema takes
    those defaults from the loader, so the two cannot differ."""
    from ds_msp.rig.calib_param import _PRODUCTION
    from ds_msp.rig.config_schema import PROPERTIES, config_schema

    prod = config_schema()["x-profiles"]["production"]
    assert prod == _PRODUCTION and prod["convergence_gate"] == "refuse"
    assert set(prod) <= set(PROPERTIES)
