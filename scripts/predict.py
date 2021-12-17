"""
To push this predictor to replicate.com, first run download_checkpoints() and save files to omnizart/checkpoints.
Then run cog push. Further documentation can be found at https://replicate.com/docs
"""

import os
import tempfile
import subprocess
import shutil
from pathlib import Path

import cog
import scipy.io.wavfile as wave

from omnizart.remote import download_large_file_from_google_drive
from omnizart.beat import app as bapp
from omnizart.chord import app as capp
from omnizart.drum import app as dapp
from omnizart.music import app as mapp
from omnizart.vocal import app as vapp
from omnizart.vocal_contour import app as vcapp


class Predictor(cog.Predictor):
    def setup(self):
        self.SF2_FILE = "general_soundfont.sf2"
        if not os.path.exists(self.SF2_FILE):
            print("Downloading soundfont...")
            download_large_file_from_google_drive(
                "16RM-dWKcNtjpBoo7DFSONpplPEg5ruvO", file_length=31277462, save_name=self.SF2_FILE
            )
        self.app = {"music": mapp, "chord": capp, "drum": dapp, "vocal": vapp, "vocal-contour": vcapp, "beat": bapp}
        self.model_path = {"piano": "Piano", "piano-v2": "PianoV2", "assemble": "Stream", "pop-song": "Pop", "": None}

    @cog.input(
        "audio",
        type=Path,
        help="Path to the input music. Supports mp3 and wav format.",
    )
    @cog.input(
        "mode",
        type=str,
        default="music-piano-v2",
        options=["music-piano", "music-piano-v2", "music-assemble", "chord", "drum", "vocal", "vocal-contour", "beat"],
        help="Transcription mode",
    )
    def predict(self, audio, mode):
        assert str(audio).endswith(".mp3") or str(audio).endswith(".wav"), "Please upload mp3 or wav file."
        temp_folder = "cog_temp"
        os.makedirs(temp_folder, exist_ok=True)
        try:
            audio_name = str(os.path.splitext(os.path.basename(audio))[0])
            if str(audio).endswith(".wav"):
                wav_file_path = str(audio)
            else:
                wav_file_path = f"{temp_folder}/{audio_name}.wav"
                subprocess.run(["ffmpeg", "-y", "-i", str(audio), wav_file_path])
            model = ""
            if mode.startswith("music"):
                mode_list = mode.split("-")
                mode = mode_list[0]
                model = "-".join(mode_list[1:])

            app = self.app[mode]
            model_path = self.model_path[model]
            midi = app.transcribe(wav_file_path, model_path=model_path)

            if mode == "vocal-contour":
                out_name = f"{audio_name}_trans.wav"
            else:
                print("Synthesizing MIDI...")
                out_name = f"{temp_folder}/{audio_name}_synth.wav"
                raw_wav = midi.fluidsynth(fs=44100, sf2_path=self.SF2_FILE)
                wave.write(out_name, 44100, raw_wav)

            out_path = Path(tempfile.mkdtemp()) / "out.mp3"  # out_path is automatically cleaned up by cog
            subprocess.run(["ffmpeg", "-y", "-i", out_name, str(out_path)])
        finally:
            shutil.rmtree(temp_folder)
            if os.path.exists(f"{audio_name}.mid"):
                os.remove(f"{audio_name}.mid")
            if os.path.exists(f"{audio_name}_trans.wav"):
                os.remove(f"{audio_name}_trans.wav")
        return out_path
