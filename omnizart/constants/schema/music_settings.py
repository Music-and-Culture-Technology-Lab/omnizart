

def simple_unit(value_type, choices=None):
    return {
        "type": "object",
        "properties": {
            "Description": {"type": "string"},
            "Value": {"type": value_type} if choices is None else {"type": value_type, "enum": choices},
            "Type": {"type": "string", "enum": ["Integer", "Float", "String", "Bool"]},
            "Choices": {"type": "array"}
        },
        "required": ["Value"],
        "additionalProperties": False
    }


def list_unit(sub_value_type, choices=None):
    return {
        "type": "object",
        "properties": {
            "Description": {"type": "string"},
            "Value": {
                "type": "array",
                "items": {"type": sub_value_type} if choices is None else {"type": sub_value_type, "enum": choices}
            },
            "Type": {"const": "List"},
            "SubType": {"type": "string", "enum": ["String", "Float", "Integer"]},
            "Choices": {"type": "array"}
        },
        "required": ["Value"],
        "additionalProperties": False
    }


FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "Description": {"type": "string"},
        "Settings": {
            "type": "object",
            "properties": {
                "HopSize": simple_unit("number"),
                "SamplingRate": simple_unit("integer"),
                "WindowSize": simple_unit("integer"),
                "FrequencyResolution": simple_unit("number"),
                "FrequencyCenter": simple_unit("number"),
                "TimeCenter": simple_unit("number"),
                "Gamma": list_unit("number"),
                "BinsPerOctave": simple_unit("integer"),
                "HarmonicNumber": simple_unit("integer"),
                "Harmonic": simple_unit("boolean"),
            },
            "required": [
                "SamplingRate",
                "HopSize",
                "WindowSize",
                "FrequencyResolution",
                "FrequencyCenter",
                "TimeCenter",
                "Gamma",
                "BinsPerOctave",
                "HarmonicNumber",
                "Harmonic"
            ],
            "additionalProperties": False
        }
    },
    "required": ["Settings"],
    "additionalProperties": False
}

DATASET_SCHEMA = {
    "type": "object",
    "properties": {
        "Description": {"type": "string"},
        "Settings": {
            "type": "object",
            "properties": {
                "SavePath": simple_unit("string"),
                "FeatureType": simple_unit("string", choices=["CFP", "HCFP"]),
                "FeatureSavePath": simple_unit("string")
            },
            "required": ["SavePath", "FeatureType", "FeatureSavePath"],
            "additionalProperties": False
        },
    },
    "required": ["Settings"],
    "additionalProperties": False
}


MODEL_CHECKPOINT_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "Description": {"type": "string"},
        "Type": {"const": "Map"},
        "SubType": {"type": "array", "items": {"type": "string"}},
        "Value": {
            "type": "object",
            "properties": {
                "Piano": {"type": "string", "format": "uri"}
            }
        }
    },
    "required": ["Value"],
    "additionalProperties": False
}

MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "Description": {"type": "string"},
        "Settings": {
            "type": "object",
            "properties": {
                "SavePrefix": simple_unit("string"),
                "SavePath": simple_unit("string"),
                "ModelType": simple_unit("string", choices=["aspp", "attn"])
            },
            "required": ["SavePrefix", "SavePath"],
            "additionalProperties": False
        }
    },
    "required": ["Settings"],
    "additionalProperties": False
}

INFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "Description": {"type": "string"},
        "Settings": {
            "type": "object",
            "properties": {
                "MinLength": simple_unit("number"),
                "InstTh": simple_unit("number"),
                "OnsetTh": simple_unit("number"),
                "DuraTh": simple_unit("number"),
                "FrameTh": simple_unit("number")
            },
            "required": ["InstTh", "OnsetTh", "DuraTh", "FrameTh"],
            "additionalProperties": False
        }
    },
    "required": ["Settings"],
    "additionalProperties": False
}

TRAINING_SCHEMA = {
    "type": "object",
    "properties": {
        "Description": {"type": "string"},
        "Settings": {
            "type": "object",
            "properties": {
                "Epoch": simple_unit("integer"),
                "Steps": simple_unit("integer"),
                "ValSteps": simple_unit("integer"),
                "BatchSize": simple_unit("integer"),
                "ValBatchSize": simple_unit("integer"),
                "EarlyStop": simple_unit("integer"),
                "LossFunction": simple_unit("string", choices=["smooth", "focal", "bce"]),
                "LabelType": simple_unit(
                    "string",
                    choices=[
                        "note-stream",
                        "frame-stream",
                        "note", "frame",
                        "true-frame",
                        "true-frame-stream",
                        "pop-note-stream"
                    ]
                ),
                "Channels": list_unit("string", choices=["Spec", "GCoS", "Ceps"]),
                "Timesteps": simple_unit("integer"),
                "FeatureNum": simple_unit("integer")
            },
            "required": [
                "Epoch",
                "Steps",
                "ValSteps",
                "BatchSize",
                "ValBatchSize",
                "EarlyStop",
                "LossFunction",
                "LabelType",
                "Channels",
                "Timesteps",
                "FeatureNum"
            ],
            "additionalProperties": False
        }
    },
    "required": ["Settings"],
    "additionalProperties": False
}

MUSIC_GENERAL_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "TranscriptionMode": simple_unit("string"),
        "CheckpointPath": MODEL_CHECKPOINT_PATH_SCHEMA,
        "Feature": FEATURE_SCHEMA,
        "Dataset": DATASET_SCHEMA,
        "Model": MODEL_SCHEMA,
        "Inference": INFERENCE_SCHEMA,
        "Training": TRAINING_SCHEMA
    },
    "required": ["TranscriptionMode", "Feature", "Dataset", "Model", "Inference", "Training"],
    "additionalProperties": False
}


MUSIC_SETTINGS_SCHEMA = {
    "$schema": "http://json-schema.org/draft/draft/2019-09/schema#",
    "type": "object",
    "properties": {
        "General": MUSIC_GENERAL_SETTINGS_SCHEMA
    },
    "required": ["General"]
}
