# Speech-to-Text Converter

A Python-based speech recognition tool that uses Mozilla's DeepSpeech engine to convert speech to text. This tool supports both real-time microphone input and audio file processing.

## Features

- Real-time speech recognition from microphone input
- Audio file conversion to text
- Easy-to-use Python class interface
- Configurable model and scorer paths
- Adjustable audio sample rate

## Prerequisites

- Python 3.8 - 3.11 (DeepSpeech 0.9.3 is not compatible with Python 3.12)
- Conda package manager
- A microphone for real-time speech recognition
- PortAudio development files (for PyAudio)

### System Dependencies

On Ubuntu/Debian, install PortAudio development files:
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
```

On Fedora/RHEL:
```bash
sudo dnf install portaudio-devel
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a conda environment:
```bash
conda create -n speech-env python=3.9
conda activate speech-env
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download DeepSpeech model files:
   ```bash
   # Create a models directory
   mkdir models
   cd models
   
   # Download the model and scorer files
   wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
   wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
   ```

   The model files will be downloaded to the `models` directory. Make sure to update the model paths when initializing the SpeechToText class:

   ```python
   stt = SpeechToText(
       model_path='models/deepspeech-0.9.3-models.pbmm',
       scorer_path='models/deepspeech-0.9.3-models.scorer'
   )
   ```

## Usage

```python
from speech_to_text import SpeechToText

# Initialize the converter
stt = SpeechToText()

# Record and convert speech from microphone
audio_data = stt.record_from_microphone(duration=5)
text = stt.convert_audio_data(audio_data)
print("Recognized text:", text)

# Convert from audio file
text = stt.convert_audio_file("path_to_your_audio.wav")
print("Recognized text from file:", text)
```

## Configuration

You can customize the following parameters when initializing the SpeechToText class:
- `model_path`: Path to DeepSpeech model file
- `scorer_path`: Path to DeepSpeech scorer file
- `sample_rate`: Audio sample rate (default: 16000Hz)

## License

[Your chosen license]
