#https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z click on this link download ffmeg , copy path of bin folder after unzipping this and set it as your system path by editting environment variables
#pip install git+https://github.com/snakers4/silero-vad.git install this as well

import torch
import subprocess
import tempfile
import os
import silero_vad

def has_voice(audio_file_path):
    """
    Simple function to detect if audio file contains human voice.
    Returns True if voice detected, False otherwise.
    """
    # Load Silero VAD model
    model = silero_vad.load_silero_vad()
    
    # Convert audio to 16kHz WAV using FFmpeg
    temp_wav = tempfile.mktemp(suffix='.wav')
    
    try:
        # Convert to required format
        subprocess.run([
            'ffmpeg', '-i', audio_file_path, 
            '-ar', '16000', '-ac', '1', '-y', temp_wav
        ], check=True, capture_output=True)
        
        # Read audio
        wav = silero_vad.read_audio(temp_wav)
        
        # Get speech timestamps
        speech_timestamps = silero_vad.get_speech_timestamps(wav, model)
        
        # Return True if any speech detected
        return len(speech_timestamps) > 0
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

# Usage example
if __name__ == "__main__":
    audio_file = "nigga.mp3"  # Replace with your file
    
    if has_voice(audio_file):
        print("✓ Voice detected!")
    else:
        print("✗ No voice detected")
