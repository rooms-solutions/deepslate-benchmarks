import asyncio
import os
import sys
import time
import wave
from pathlib import Path

import librosa
import numpy as np
import websockets
from datasets import load_dataset

# Ensure repo-root modules are importable when running from subdirectories.
BENCH_DIR = Path(__file__).resolve().parent
ROOT_DIR = BENCH_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import realtime_pb2

# --- CONFIGURATION ---
API_KEY = os.getenv("DEEPSLATE_API_KEY")
VENDOR_ID = os.getenv("DEEPSLATE_VENDOR_ID")
ORG_ID = os.getenv("DEEPSLATE_ORG_ID")
WS_URL = f"wss://app.deepslate.eu/api/v1/vendors/{VENDOR_ID}/organizations/{ORG_ID}/realtime"
OUTPUT_DIR = BENCH_DIR / "benchmark_outputs"
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
ELEVEN_LABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

os.makedirs(OUTPUT_DIR, exist_ok=True)


async def run_single_inference(audio_array, sample_rate, question_id, request_start):
    """
    Sends one audio question to Deepslate and captures the audio response.
    """
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Resample audio to 48kHz Mono for Deepslate
    if sample_rate != 48000:
        audio_array = librosa.resample(y=audio_array, orig_sr=sample_rate, target_sr=48000)

    # Convert float32 [-1, 1] to int16 PCM
    pcm_data = (audio_array * 32767).astype(np.int16).tobytes()

    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        # Initialize Session
        init_req = realtime_pb2.ServiceBoundMessage()

        # Audio Input Setup
        init_req.initialize_session_request.input_audio_line.sample_rate = 48000
        init_req.initialize_session_request.input_audio_line.channel_count = 1
        init_req.initialize_session_request.input_audio_line.sample_format = 1

        # Audio Output Setup
        init_req.initialize_session_request.output_audio_line.sample_rate = 16000
        init_req.initialize_session_request.output_audio_line.channel_count = 1
        init_req.initialize_session_request.output_audio_line.sample_format = 1

        # VAD Configuration
        init_req.initialize_session_request.vad_configuration.confidence_threshold = 0.5
        init_req.initialize_session_request.vad_configuration.min_volume = 0.0
        init_req.initialize_session_request.vad_configuration.start_duration.nanos = 300000000  # 300ms
        init_req.initialize_session_request.vad_configuration.stop_duration.nanos = 700000000  # 700ms
        init_req.initialize_session_request.vad_configuration.backbuffer_duration.seconds = 1

        # Inference Config
        init_req.initialize_session_request.inference_configuration.system_prompt = (
            "You are a helpful assistant."
        )

        # TTS CONFIGURATION
        init_req.initialize_session_request.tts_configuration.eleven_labs.api_key = ELEVEN_LABS_API_KEY
        init_req.initialize_session_request.tts_configuration.eleven_labs.voice_id = ELEVEN_LABS_VOICE_ID

        init_payload = init_req.SerializeToString()
        await ws.send(init_payload)

        # Stream Audio Chunks
        chunk_size = 9600  # 100ms chunks at 48kHz
        packet_id = 0

        chunk_start = time.monotonic()
        last_real_audio_send_time = None
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            msg = realtime_pb2.ServiceBoundMessage()

            msg.user_input.packet_id = packet_id
            msg.user_input.mode = realtime_pb2.InferenceTriggerMode.IMMEDIATE
            msg.user_input.audio_data.data = chunk

            msg_payload = msg.SerializeToString()
            await ws.send(msg_payload)
            last_real_audio_send_time = time.monotonic()
            packet_id += 1
            # Pace sending to real-time audio duration.
            chunk_duration = len(chunk) / (2 * 48000)
            target_elapsed = packet_id * chunk_duration
            elapsed = time.monotonic() - chunk_start
            sleep_for = target_elapsed - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

        # Receive Response
        audio_chunks = []
        response_complete = False

        async def send_silence_until_response():
            nonlocal packet_id
            silence_chunk = b"\x00" * chunk_size
            silence_duration = len(silence_chunk) / (2 * 48000)
            while not response_complete:
                silence_msg = realtime_pb2.ServiceBoundMessage()
                silence_msg.user_input.packet_id = packet_id
                silence_msg.user_input.mode = realtime_pb2.InferenceTriggerMode.NO_TRIGGER
                silence_msg.user_input.audio_data.data = silence_chunk
                await ws.send(silence_msg.SerializeToString())
                packet_id += 1
                await asyncio.sleep(silence_duration)

        silence_task = asyncio.create_task(send_silence_until_response())

        while not response_complete:
            try:
                response_data = await asyncio.wait_for(ws.recv(), timeout=500.0)
                response = realtime_pb2.ClientBoundMessage()
                response.ParseFromString(response_data)

                if response.HasField('playback_clear_buffer'):
                    audio_chunks = []

                if response.HasField('model_audio_chunk'):
                    audio_chunks.append(response.model_audio_chunk.audio.data)

                if response.HasField('response_end'):
                    response_complete = True

            except asyncio.TimeoutError:
                response_complete = True
                silence_task.cancel()
                try:
                    await silence_task
                except asyncio.CancelledError:
                    pass
                print(f"Question {question_id}: Timeout waiting for response.")
                return None

        try:
            await silence_task
        except asyncio.CancelledError:
            pass
        return b"".join(audio_chunks)


def main():
    # Load dataset
    print("Loading Big Bench Audio...")
    dataset = load_dataset("ArtificialAnalysis/big_bench_audio", split="train")

    # Run Loop
    for i, item in enumerate(dataset):
        question_id = item['id']
        audio_array = item['audio']['array']
        sr = item['audio']['sampling_rate']

        output_filename = OUTPUT_DIR / f"response_{question_id}.wav"

        print(f"Processing {i}/{len(dataset)} (ID: {question_id})...")
        request_audio_duration = len(audio_array) / sr
        print(
            f"Question {question_id}: Request audio duration "
            f"{request_audio_duration:.2f}s"
        )

        # Run async inference
        request_start = time.monotonic()
        response_audio = asyncio.run(run_single_inference(audio_array, sr, question_id, request_start))

        if response_audio is None:
            print(f"Skipping save for {question_id} due to timeout.")
            continue

        if not response_audio:
            print(f"Skipping save for {question_id} due to empty response.")
            continue

        with wave.open(os.fspath(output_filename), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(response_audio)


if __name__ == "__main__":
    main()
