import whisper
import sounddevice as sd
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

# ==================== تنظیمات ====================
MODEL_SIZE = "small"  # یا "medium" برای دقت بیشتر
SAMPLERATE = 16000  # ⚠️ حتماً 16000 باشد (32000 باعث خطا می‌شود!)
CHUNK_DURATION = 1.0
MIN_SPEECH_DURATION = 0.3
OVERLAP = 0.3

# ==================== بارگذاری مدل‌ها ====================
print("در حال بارگذاری مدل‌ها...")
vad_model = load_silero_vad()
whisper_model = whisper.load_model(MODEL_SIZE)
print(f"✓ مدل Whisper ({MODEL_SIZE}) و VAD بارگذاری شدند")

# ==================== ضبط و پردازش هوشمند ====================
buffer = np.array([], dtype=np.float32)
print("\n🎤 شروع ضبط... (Ctrl+C برای توقف)")

try:
    while True:
        audio_chunk = sd.rec(
            int(CHUNK_DURATION * SAMPLERATE),
            samplerate=SAMPLERATE,  # ✅ 16000
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        audio_chunk = audio_chunk.flatten()

        if len(buffer) > 0:
            audio_chunk = np.concatenate([buffer, audio_chunk])
        buffer = audio_chunk[-int(OVERLAP * SAMPLERATE):]

        # ✅ حالا با 16000 هرتز کار می‌کند
        speech_timestamps = get_speech_timestamps(
            audio_chunk,
            vad_model,
            sampling_rate=SAMPLERATE,  # ✅ 16000
            min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
            threshold=0.5
        )

        if speech_timestamps:
            print("\n🔊 گفتار شناسایی شد...")
            speech_segments = [audio_chunk[ts['start']:ts['end']] for ts in speech_timestamps]
            full_speech = np.concatenate(speech_segments)
            full_speech = full_speech / (np.max(np.abs(full_speech)) + 1e-8)

            # ✅ بدون تعیین زبان — Whisper خودش فارسی/انگلیسی را تشخیص می‌دهد
            result = whisper_model.transcribe(
                full_speech,
                language='en',
                fp16=torch.cuda.is_available(),
                temperature=0.0,
                no_speech_threshold=0.4
            )

            text = result["text"].strip()
            if len(text) > 3:
                print(f"💬 {text}")
            else:
                print("⚠️ متن بسیار کوتاه (احتمالاً نویز)")

except KeyboardInterrupt:
    print("\n⏹️ ضبط متوقف شد")
except Exception as e:
    print(f"\n❌ خطا: {type(e).__name__} - {e}")
    print("💡 نکته: اگر خطا مربوط به نرخ نمونه‌برداری است، حتماً SAMPLERATE=16000 را بررسی کنید!")