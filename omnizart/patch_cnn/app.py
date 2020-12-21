import os
from os.path import join as jpath

import h5py

from omnizart.io import write_yaml
from omnizart.utils import get_logger, parallel_generator, get_filename
from omnizart.base import BaseTranscription
from omnizart.constants import datasets as d_struct
from omnizart.feature.cfp import extract_patch_cfp
from omnizart.setting_loaders import PatchCNNSettings
from omnizart.models.patch_cnn import patch_cnn_model


logger = get_logger("Patch CNN Transcription")


class PatchCNNTranscription(BaseTranscription):
    def __init__(self, conf_path=None):
        super().__init__(PatchCNNSettings, conf_path=conf_path)

    def transcribe(self, input_audio, model_path, output="./"):
        pass

    def generate_feature(self, dataset_path, patch_cnn_settings=None, num_threads=4):
        settings = self._validate_and_get_settings(patch_cnn_settings)

        struct = d_struct.MIR1KStructure

        ## Below are examples of dealing with multiple supported datasets.
        # dataset_type = resolve_dataset_type(
        #     dataset_path,
        #     keywords={"maps": "maps", "musicnet": "musicnet", "maestro": "maestro", "rhythm": "pop", "pop": "pop"}
        # )
        # if dataset_type is None:
        #     logger.warning(
        #         "The given path %s does not match any built-in processable dataset. Do nothing...",
        #         dataset_path
        #     )
        #     return
        # logger.info("Inferred dataset type: %s", dataset_type)
        # # Build instance mapping
        # struct = {
        #     "maps": d_struct.MapsStructure,
        #     "musicnet": d_struct.MusicNetStructure,
        #     "maestro": d_struct.MaestroStructure,
        #     "pop": d_struct.PopStructure
        # }[dataset_type]
        # label_extractor = {
        #     "maps": MapsLabelExtraction,
        #     "musicnet": MusicNetLabelExtraction,
        #     "maestro": MaestroLabelExtraction,
        #     "pop": PopLabelExtraction
        # }[dataset_type]

        # Fetching wav files
        train_data_pair = struct.get_train_data_pair(dataset_path=dataset_path)
        test_data_pair = struct.get_test_data_pair(dataset_path=dataset_path)
        logger.info("Number of total training wavs: %d", len(train_data_pair))
        logger.info("Number of total testing wavs: %d", len(test_data_pair))

        # Resolve feature output path
        train_feat_out_path, test_feat_out_path = self._resolve_feature_output_path(dataset_path, settings)
        logger.info("Output training feature to %s", train_feat_out_path)
        logger.info("Output testing feature to %s", test_feat_out_path)

        # Feature extraction
        logger.info(
            "Start extracting the training feature. "
            "This may take time to finish and affect the computer's performance"
        )
        _parallel_feature_extraction(
            train_data_pair, out_path=train_feat_out_path, feat_settings=settings.feature, num_threads=num_threads
        )
        logger.info(
            "Start extracting the testing feature. "
            "This may take time to finish and affect the computer's performance"
        )
        _parallel_feature_extraction(
            test_data_pair, out_path=test_feat_out_path, feat_settings=settings.feature, num_threads=num_threads
        )

        # Writing out the settings
        write_yaml(settings.to_json(), jpath(train_feat_out_path, ".success.yaml"))
        write_yaml(settings.to_json(), jpath(test_feat_out_path, ".success.yaml"))
        logger.info("All done")

    def train(self, feature_folder, model_name=None, input_model_path=None, patch_cnn_settings=None):
        pass


def _all_in_one_extract(data_pair, **feat_params):
    feat, mapping, zzz = extract_patch_cfp(data_pair[0], **feat_params)

    # TODO: implement label extraction and execute here
    return feat, mapping, zzz


def _parallel_feature_extraction(data_pair_list, out_path, feat_settings, num_threads=4):
    feat_params = {
        "patch_size": feat_settings.patch_size,
        "threshold": feat_settings.peak_threshold,
        "down_fs": feat_settings.sampling_rate,
        "hop": feat_settings.hop_size,
        "win_size": feat_settings.window_size,
        "fr": feat_settings.frequency_resolution,
        "fc": feat_settings.frequency_center,
        "tc": feat_settings.time_center,
        "g": feat_settings.gamma,
        "bin_per_octave": feat_settings.bins_per_octave,
    }

    iters = enumerate(
        parallel_generator(
            _all_in_one_extract,
            data_pair_list,
            max_workers=num_threads,
            use_thread=True,
            chunk_size=num_threads,
            **feat_params
        )
    )
    for idx, ((feat, mapping, zzz), audio_idx) in iters:
        audio = data_pair_list[audio_idx][0]

        # logger.info("Progress: %s/%s - %s", idx+1, len(data_pair_list), audio)
        print(f"Progress: {idx + 1}/{len(data_pair_list)} - {audio}", end="\r")

        filename = get_filename(audio)
        out_hdf = jpath(out_path, filename + ".hdf")
        with h5py.File(out_hdf, "w") as out_f:
            out_f.create_dataset("feature", data=feat)
            out_f.create_dataset("mapping", data=mapping)
            out_f.create_dataset("Z", data=zzz)
    print("")


if __name__ == "__main__":
    app = PatchCNNTranscription()
    app.generate_feature("/data/MIR-1K")
