import deepspeech
import numpy as np
import wave
import pyaudio
import time
import webrtcvad
import collections

class SpeechToText:
    def __init__(self, model_path='models/deepspeech-0.9.3-models.pbmm', 
                 scorer_path='models/deepspeech-0.9.3-models.scorer',
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
        
    def record_from_microphone(self, max_duration=60, vad_aggressiveness=3, silence_duration=3.0):
        """
        Record audio from microphone with VAD-based stopping
        
        Args:
            max_duration (int): Maximum recording duration in seconds
            vad_aggressiveness (int): VAD aggressiveness (0-3)
            silence_duration (float): Duration of silence to stop recording
            
        Returns:
            numpy.ndarray: Recorded audio data
        """
        # Initialize VAD
        vad = webrtcvad.Vad(vad_aggressiveness)
        frame_duration_ms = 30  # WebRTC works with 10, 20, or 30ms frames
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        
        # Configure audio stream
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=frame_size
        )
        
        print("Recording... (speak to begin)")
        frames = []
        silent_frames = 0
        max_silent_frames = int(silence_duration * 1000 / frame_duration_ms)
        start_time = time.time()
        
        # Use a ring buffer to track if we've started speaking
        ring_buffer = collections.deque(maxlen=max_silent_frames)
        recording_started = False
        
        while True:
            if time.time() - start_time > max_duration:
                print("\nMaximum duration reached")
                break
                
            frame = stream.read(frame_size)
            is_speech = vad.is_speech(frame, self.sample_rate)
            print(f"Speech detected: {is_speech} | Silent frames: {silent_frames}/{max_silent_frames}", end='\r')
            
            if not recording_started:
                ring_buffer.append(frame)
                num_voiced = len([f for f in range(len(ring_buffer)) if vad.is_speech(ring_buffer[f], self.sample_rate)])
                if num_voiced > 0.5 * ring_buffer.maxlen:
                    recording_started = True
                    print("\nSpeech detected, recording...")
                    frames.extend(list(ring_buffer))
                continue
            
            frames.append(frame)
            
            if is_speech:
                silent_frames = 0
            else:
                silent_frames += 1
                
            if silent_frames >= max_silent_frames and recording_started:
                print("\nSilence detected, stopping recording")
                break
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        if not frames:
            return np.array([], dtype=np.int16)
            
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
    audio_data = stt.record_from_microphone(max_duration=60)
    
    # Convert to text
    text = stt.convert_audio_data(audio_data)
    print("Recognized text:", text)
    
    # Example with audio file
    # text = stt.convert_audio_file("path_to_your_audio.wav")
    # print("Recognized text from file:", text)
