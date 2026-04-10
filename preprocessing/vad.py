import numpy as np

from preprocessing.chunking import cut_and_merge_segments


def get_vad_segments(
    audio_array,
    sample_rate=16000,
    frame_ms=30,
    hop_ms=10,
    energy_threshold_ratio=0.5,
    min_speech_ms=500,
    min_silence_ms=200,
    max_chunk_seconds=30.0,
):
    """
    Simple energy-based VAD.

    Returns:
        [{"start": float, "end": float}, ...]

    Notes:
    - This is a lightweight fallback VAD with no heavy dependencies.
    - It works best as a basic speech/silence detector, not as a perfect VAD.
    """
    if len(audio_array) == 0:
        return []

    audio = np.asarray(audio_array, dtype=np.float32)

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    min_speech_frames = max(1, int(min_speech_ms / hop_ms))
    min_silence_frames = max(1, int(min_silence_ms / hop_ms))

    energies = []
    frame_starts = []

    for start in range(0, max(1, len(audio) - frame_len + 1), hop_len):
        frame = audio[start:start + frame_len]
        energy = float(np.sqrt(np.mean(frame ** 2) + 1e-10))
        energies.append(energy)
        frame_starts.append(start)

    if not energies:
        duration = len(audio) / float(sample_rate)
        return [{"start": 0.0, "end": duration}]

    energies = np.asarray(energies, dtype=np.float32)

    baseline = float(np.percentile(energies, 20))
    peak = float(np.percentile(energies, 95))
    threshold = baseline + energy_threshold_ratio * max(peak - baseline, 1e-8)

    speech_mask = energies > threshold

    raw_segments = []
    in_speech = False
    speech_start_idx = 0
    silence_count = 0

    for i, is_speech in enumerate(speech_mask):
        if is_speech:
            if not in_speech:
                in_speech = True
                speech_start_idx = i
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    speech_end_idx = i - silence_count + 1
                    if (speech_end_idx - speech_start_idx) >= min_speech_frames:
                        start_sec = frame_starts[speech_start_idx] / sample_rate
                        end_sample = min(
                            len(audio),
                            frame_starts[speech_end_idx] + frame_len,
                        )
                        end_sec = end_sample / sample_rate
                        raw_segments.append({"start": start_sec, "end": end_sec})
                    in_speech = False
                    silence_count = 0

    if in_speech:
        speech_end_idx = len(speech_mask) - 1
        if (speech_end_idx - speech_start_idx + 1) >= min_speech_frames:
            start_sec = frame_starts[speech_start_idx] / sample_rate
            end_sample = min(len(audio), frame_starts[speech_end_idx] + frame_len)
            end_sec = end_sample / sample_rate
            raw_segments.append({"start": start_sec, "end": end_sec})

    if not raw_segments:
        duration = len(audio) / float(sample_rate)
        return [{"start": 0.0, "end": duration}]

    return cut_and_merge_segments(
        raw_segments,
        max_chunk_seconds=max_chunk_seconds,
        max_gap_seconds=0.5,
    )