from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import os

class AudioUtils:
    @staticmethod
    def record_audio(transcriber, chunk_length_s, stream_chunk_s, save_to_file=None):
        """
        Record audio using a live microphone stream.

        :param transcriber: Transcriber object to get sampling rate.
        :param chunk_length_s: Length of audio chunks in seconds.
        :param stream_chunk_s: Stream chunk duration in seconds.
        :param save_to_file: Optional path to save the recorded audio.
        :return: Recorded audio data as a generator.
        """
        sampling_rate = transcriber.feature_extractor.sampling_rate
        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )

        if save_to_file:
            with sf.SoundFile(save_to_file, mode='w', samplerate=sampling_rate, channels=1) as file:
                for chunk in mic:
                    file.write(chunk)
        return mic

    @staticmethod
    def play_audio(audio_data, samplerate, format="wav", volume=1.0):
        """
        Play audio data with optional volume control.

        :param audio_data: NumPy array containing the audio waveform.
        :param samplerate: Sampling rate of the audio.
        :param format: Audio format for temporary file (default: 'wav').
        :param volume: Playback volume (default: 1.0).
        """
        temp_file = f"temp_audio.{format}"
        sf.write(temp_file, audio_data * volume, samplerate)
        song = AudioSegment.from_file(temp_file, format=format)
        play(song)
        os.remove(temp_file)

    @staticmethod
    def visualize_audio(audio_data, samplerate):
        """
        Plot the waveform of the audio data.

        :param audio_data: NumPy array containing the audio waveform.
        :param samplerate: Sampling rate of the audio.
        """
        time_axis = np.linspace(0, len(audio_data) / samplerate, num=len(audio_data))
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, audio_data, label="Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Audio Waveform")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def process_audio_in_real_time(callback, duration, samplerate=16000):
        """
        Capture and process audio in real-time with a callback.

        :param callback: Function to process audio chunks in real-time.
        :param duration: Total duration of the recording in seconds.
        :param samplerate: Sampling rate of the audio.
        """
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            callback(indata)

        with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
            sd.sleep(int(duration * 1000))

# Example usage:
# Visualize a waveform
# audio_data = np.random.rand(16000)  # Replace with actual audio data
# AudioUtils.visualize_audio(audio_data, samplerate=16000)

# Record and save to file
# mic_stream = AudioUtils.record_audio(transcriber, 5, 1, save_to_file="output.wav")
