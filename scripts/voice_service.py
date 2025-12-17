#!/usr/bin/env python3
"""
Voice Command Service using Whisper (OpenAI) - Much more accurate than vosk!

Runs on HOST (outside Docker). Captures audio, recognizes speech with Whisper,
and publishes commands to Redis.

Usage:
  python voice_service_whisper.py [--model tiny/base/small] [--redis-host HOST]

Requirements:
  pip install faster-whisper sounddevice redis numpy

Models (accuracy vs speed):
  tiny  - fastest, ~1GB VRAM, decent accuracy
  base  - good balance, ~1GB VRAM
  small - better accuracy, ~2GB VRAM
  medium - high accuracy, ~5GB VRAM (if you have the VRAM)
"""

import argparse
import json
import sys
import time
import numpy as np
import queue
import threading

try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
    import redis
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install faster-whisper sounddevice redis numpy")
    sys.exit(1)


class WhisperVoiceService:
    """Voice command service using Whisper for accurate recognition"""
    
    # Command words - Whisper is accurate enough to use normal words!
    STOP_WORDS = ['stop', 'quit', 'exit', 'terminate', 'shutdown', 'kill']  # Emergency stop
    PAUSE_WORDS = ['pause', 'freeze', 'halt', 'wait', 'hold']  # Pause/freeze
    RESUME_WORDS = ['resume', 'go', 'start', 'continue', 'run', 'move', 'yes']
    
    def __init__(self, model_size='base', redis_host='localhost', redis_port=6379, use_gpu=False):
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 2.0  # seconds of audio per recognition
        self.audio_queue = queue.Queue()
        
        # Debounce - prevent duplicate commands
        self.last_command = None
        self.last_command_time = 0
        self.command_cooldown = 2.0  # seconds before same command can be sent again
        
        # Connect to Redis
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        try:
            self.redis_client.ping()
            print(f"✓ Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError:
            print(f"✗ Cannot connect to Redis at {redis_host}:{redis_port}")
            sys.exit(1)
        
        # Load Whisper model
        print(f"Loading Whisper model '{model_size}'... (first run downloads ~150MB-1GB)")
        if use_gpu:
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print(f"✓ Whisper model '{model_size}' loaded (GPU)")
        else:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print(f"✓ Whisper model '{model_size}' loaded (CPU)")
        
        # Get audio device
        self.device_info = sd.query_devices(kind='input')
        print(f"✓ Using microphone: {self.device_info['name']}")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def publish_command(self, command, text):
        """Publish command to Redis with debounce"""
        now = time.time()
        
        # Debounce: skip if same command within cooldown period
        if command == self.last_command and (now - self.last_command_time) < self.command_cooldown:
            return False
        
        self.last_command = command
        self.last_command_time = now
        
        msg = {
            'command': command,
            'text': text,
            'timestamp': now
        }
        self.redis_client.publish('voice_commands', json.dumps(msg))
        self.redis_client.set('voice_command', json.dumps(msg))
        self.redis_client.expire('voice_command', 5)
        print(f"📢 Published: {command.upper()} ('{text}')")
        return True
    
    def process_text(self, text):
        """Check if text contains a command"""
        text_lower = text.lower().strip()
        
        # Check STOP first (highest priority - emergency stop)
        for word in self.STOP_WORDS:
            if word in text_lower:
                return 'stop', text
        
        # Then PAUSE
        for word in self.PAUSE_WORDS:
            if word in text_lower:
                return 'pause', text
        
        # Then RESUME
        for word in self.RESUME_WORDS:
            if word in text_lower:
                return 'resume', text
        
        return None, text
    
    def run(self):
        """Main loop"""
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        print("\n" + "="*50)
        print("🎤 WHISPER VOICE SERVICE RUNNING")
        print("="*50)
        print("Say any of these words naturally:")
        print(f"  🛑 STOP:   {', '.join(self.STOP_WORDS)}")
        print(f"  ⏸️  PAUSE:  {', '.join(self.PAUSE_WORDS)}")
        print(f"  ▶️  RESUME: {', '.join(self.RESUME_WORDS)}")
        print("")
        print("💡 Whisper is much more accurate than vosk!")
        print("Press Ctrl+C to stop this service")
        print("="*50 + "\n")
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                               dtype='float32', callback=self.audio_callback,
                               blocksize=chunk_samples):
                
                audio_buffer = np.array([], dtype=np.float32)
                
                while True:
                    # Collect audio chunks
                    while not self.audio_queue.empty():
                        chunk = self.audio_queue.get()
                        audio_buffer = np.append(audio_buffer, chunk.flatten())
                    
                    # Process when we have enough audio
                    if len(audio_buffer) >= chunk_samples:
                        # Take chunk and clear buffer (no overlap to prevent duplicates)
                        audio_chunk = audio_buffer[:chunk_samples]
                        audio_buffer = np.array([], dtype=np.float32)
                        
                        # Skip if too quiet (silence)
                        if np.abs(audio_chunk).max() < 0.01:
                            continue
                        
                        # Transcribe with Whisper
                        segments, info = self.model.transcribe(
                            audio_chunk, 
                            language='en',
                            vad_filter=True,  # Filter out silence
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                        
                        for segment in segments:
                            text = segment.text.strip()
                            if text and len(text) > 1:
                                command, _ = self.process_text(text)
                                if command:
                                    self.publish_command(command, text)
                                else:
                                    print(f"💬 Heard: '{text}'")
                    
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Voice service stopped")


def main():
    parser = argparse.ArgumentParser(description="Whisper voice command service")
    parser.add_argument('--model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: base)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU (requires cuDNN)')
    parser.add_argument('--redis-host', type=str, default='localhost')
    parser.add_argument('--redis-port', type=int, default=6379)
    
    args = parser.parse_args()
    
    service = WhisperVoiceService(args.model, args.redis_host, args.redis_port, use_gpu=args.gpu)
    service.run()


if __name__ == '__main__':
    main()
