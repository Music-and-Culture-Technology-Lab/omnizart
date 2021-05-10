import os
import shutil

import pytest
import numpy as np
from jsonschema import ValidationError

from omnizart import utils
from omnizart import io


def test_logger(caplog):
    logger = utils.get_logger("MyLogger")
    logger.info("hello")
    for record in caplog.records:
        assert record.levelname == "INFO"
        assert "hello" in record.message

    logger = utils.get_logger("MyLogger")
    assert len(logger.handlers) == 1


def test_pickle_io():
    data = {
        "int": 1,
        "list": [1,2,3],
        "dict": {"a": "haha", "b": "iii"},
        "rrrr": "RRRRRRRRRRRRRRRRRR!"
    }
    path = "./tmp/"
    f_name = os.path.join(path, "one_more_level", "test.pickle")
    io.dump_pickle(data, f_name)
    loaded = io.load_pickle(f_name)
    assert data == loaded
    shutil.rmtree(path)


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
    io.write_yaml(data, f_name)
    loaded = io.load_yaml(f_name)
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


def test_load_audio_with_librosa():
    audio = "./tests/resource/sample.wav"
    samp = 16000
    data, sampling_rate = io.load_audio_with_librosa(audio, sampling_rate=samp)
    assert sampling_rate == samp
    assert len(data) == 749252


def test_load_audio():
    audio = "./tests/resource/sample.wav"
    data, fs = io.load_audio(audio, sampling_rate=44100, mono=False)
    assert fs == 44100
    assert data.shape == (2065124, 2)


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


def test_serializable_fail_to_match_attributes():
    data_a = DataA()
    data = {"A": 20, "B": "bbbbb"}
    with pytest.raises(AttributeError) as exc:
        data_a.from_json(data)
        assert "Attribute C is not defined in configuration file for class DataA" in exc


def test_serializable_recursive_value_path():
    data_a = DataA()
    data_a.value_path = "./level1/level2"
    data = {
        "A": {"Level1": {"Level2": 10}},
        "B": {"Level1": {"Level2": None}},
        "C": {"Level1": {"Level2": "HelloWorld"}}
    }
    assert data_a.to_json() == data


def test_aggregate_f0_info():
    t_unit = 0.01
    data = np.array([0, 0, 0, 440, 440, 440, 440, 0, 0, 0, 220, 220])
    expected = [
        {"start_time": 0.03, "end_time": 0.07, "frequency": 440, "pitch": 69},
        {"start_time": 0.1, "end_time": 0.12, "frequency": 220, "pitch": 57}
    ]
    results = utils.aggregate_f0_info(data, t_unit)
    assert results == expected

    output_path = "result.tmp"
    io.write_agg_f0_results(results, output_path)

    # Invalid column
    with pytest.raises(ValueError):
        results[0]["invalid"] = ""
        io.write_agg_f0_results(results, output_path)

    with pytest.raises(ValueError):
        del results[0]["invalid"]
        del results[0]["pitch"]
        io.write_agg_f0_results(results, output_path)

    os.remove(output_path)
