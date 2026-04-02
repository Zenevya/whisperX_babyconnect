SAMPLE_RATE = 16000


def build_fixed_chunks(duration_seconds, chunk_size=30.0):
    chunks = []
    start = 0.0

    while start < duration_seconds:
        end = min(start + chunk_size, duration_seconds)
        chunks.append({"start": start, "end": end})
        start = end

    return chunks


def slice_audio(audio_array, start_sec, end_sec, sample_rate=SAMPLE_RATE):
    start_idx = int(start_sec * sample_rate)
    end_idx = int(end_sec * sample_rate)
    return audio_array[start_idx:end_idx]


def split_long_segments(segments, max_chunk_seconds=30.0):
    split_segments = []

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])

        while (end - start) > max_chunk_seconds:
            split_segments.append(
                {
                    "start": start,
                    "end": start + max_chunk_seconds,
                }
            )
            start += max_chunk_seconds

        split_segments.append({"start": start, "end": end})

    return split_segments


def merge_close_segments(segments, max_chunk_seconds=30.0, max_gap_seconds=0.5):
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        gap = float(seg["start"]) - float(prev["end"])
        new_duration = float(seg["end"]) - float(prev["start"])

        if gap <= max_gap_seconds and new_duration <= max_chunk_seconds:
            prev["end"] = float(seg["end"])
        else:
            merged.append(seg.copy())

    return merged


def cut_and_merge_segments(
    segments,
    max_chunk_seconds=30.0,
    max_gap_seconds=0.5,
):
    segments = split_long_segments(segments, max_chunk_seconds=max_chunk_seconds)
    segments = merge_close_segments(
        segments,
        max_chunk_seconds=max_chunk_seconds,
        max_gap_seconds=max_gap_seconds,
    )
    return segments