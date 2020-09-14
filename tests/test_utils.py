import os

import pytest
from jsonschema import ValidationError

from omnizart import utils


def test_pickle_io(tmp_path):
    data = {
        "int": 1,
        "list": [1,2,3],
        "dict": {"a": "haha", "b": "iii"},
        "rrrr": "RRRRRRRRRRRRRRRRRR!"
    }

    f_name = tmp_path.joinpath("test.pickle")
    utils.dump_pickle(data, f_name)
    loaded = utils.load_pickle(f_name)

    assert data == loaded


def test_yaml_io(tmp_path):
    data = {
        "int": 1,
        "list": [1,2,3],
        "dict": {"a": "haha", "b": "iii"},
        "rrrr": "RRRRRRRRRRRRRRRRRR!",
        "nested": {
            "one": [3,4,5,6.1,9],
            "two": {"aa": {"onon": 0.04, "ofof": 0.16}, "bb": "world"}
        }
    }
    f_name = tmp_path.joinpath("test.yaml")
    utils.write_yaml(data, f_name)
    loaded = utils.load_yaml(f_name)
    assert data == loaded


@pytest.mark.parametrize("name,expected", [
    ("hello", "Hello"),
    ("normal_case_haha", "NormalCaseHaha"),
    ("aLittle_bit__strange__", "AlittleBitStrange"),
    ("__wHat_th_EF__iS____thi___S", "WhatThEfIsThiS")
])
def test_snake_to_camel(name, expected):
    assert utils.snake_to_camel(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("World", "world"),
    ("NormalCase", "normal_case"),
    ("oneMoreChance", "one_more_chance"),
    ("_LetsCrack_Regular", "_lets_crack_regular")
])
def test_camel_to_snake(name, expected):
    assert utils.camel_to_snake(name) == expected


@utils.json_serializable()
class DataA:
    invisible = "You cant't see me"
    def __init__(self):
        self.a = 10
        self.b = None
        self.c = "HelloWorld"


@utils.json_serializable(key_path="./settings", value_path="./value")
class DataB:
    def __init__(self, schema=None):
        self.schema = schema  # Reserved word of the serializable object.
        self.nested = DataA()
        self.d = [1,2,3,4]


def test_normal_serializable():
    data_a = DataA()
    expected_a = {"A": 10, "B": None, "C": "HelloWorld"}
    assert data_a.to_json() == expected_a

    data_b = DataB()
    expected_b = {
        "Settings": {
            "D": {"Value": [1, 2, 3, 4]},
            "Nested": expected_a
        }
    }
    assert data_b.to_json() == expected_b

    input_data = {
        "Settings": {
            "D": {"Value": [100, 200, 300]},
            "Nested": {
                "A": "Now I'm a string",
                "B": "I got a value!!",
                "C": {"Born": 1, "Life": 0.999}
            }
        }
    }
    assert data_b.from_json(input_data).to_json() == input_data


def test_serializable_with_schema():
    schema = {
        "$schema": "http://json-schema.org/draft/2019-09/schema#",
        "type": "object",
        "properties": {
            "Settings": {
                "type": "object",
                "properties": {
                    "D": {
                        "type": "object", 
                        "properties": {
                            "Value": {"type": "array", "items": {"type": "integer"}}
                        }
                    },
                    "Nested": {
                        "type": "object",
                        "properties": {
                            "A": {"type": "integer"},
                            "B": {"type": "null"},
                            "C": {"type": "string"}
                        }
                    }
                }
            }
        }
    }

    data_b = DataB(schema=schema)
    example_input = data_b.to_json()
    data_b.from_json(example_input)

    example_input["Settings"]["Nested"]["B"] = "Some non-null value"
    with pytest.raises(ValidationError):
        data_b.from_json(example_input)
