import deepspeech
import numpy as np
import wave
import pyaudio
import time

class SpeechToText:
    def __init__(self, model_path='deepspeech-0.9.3-models.pbmm', 
                 scorer_path='deepspeech-0.9.3-models.scorer',
                 sample_rate=16000):
        """
        Initialize the speech to text converter
        
        Args:
            model_path (str): Path to DeepSpeech model file
            scorer_path (str): Path to DeepSpeech scorer file
            sample_rate (int): Audio sample rate, default 16000Hz
        """
        self.sample_rate = sample_rate
        self.model = deepspeech.Model(model_path)
        self.model.enableExternalScorer(scorer_path)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
    def __del__(self):
        """Cleanup PyAudio"""
        self.audio.terminate()
        
    def record_from_microphone(self, duration=5):
        """
        Record audio from microphone
        
        Args:
            duration (int): Recording duration in seconds
            
        Returns:
            numpy.ndarray: Recorded audio data
        """
        # Configure audio stream
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        print("Recording...")
        frames = []
        
        # Record audio for specified duration
        for i in range(0, int(self.sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        
        print("Recording finished")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Convert audio frames to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        return audio_data
    
    def convert_audio_file(self, audio_file_path):
        """
        Convert audio file to text
        
        Args:
            audio_file_path (str): Path to audio file
            
        Returns:
            str: Recognized text
        """
        with wave.open(audio_file_path, 'rb') as w:
            rate = w.getframerate()
            frames = w.getnframes()
            buffer = w.readframes(frames)
            
        audio = np.frombuffer(buffer, dtype=np.int16)
        return self.model.stt(audio)
    
    def convert_audio_data(self, audio_data):
        """
        Convert audio data to text
        
        Args:
            audio_data (numpy.ndarray): Audio data
            
        Returns:
            str: Recognized text
        """
        return self.model.stt(audio_data)

if __name__ == "__main__":
    # Create instance of SpeechToText
    stt = SpeechToText()
    
    # Record from microphone
    audio_data = stt.record_from_microphone(duration=5)
    
    # Convert to text
    text = stt.convert_audio_data(audio_data)
    print("Recognized text:", text)
    
    # Example with audio file
    # text = stt.convert_audio_file("path_to_your_audio.wav")
    # print("Recognized text from file:", text)
