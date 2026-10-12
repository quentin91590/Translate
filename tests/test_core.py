import subprocess
from pathlib import Path

import pytest

from transcriber import audio, formats
from transcriber.backends.qwen import split_timed
from transcriber.models import Segment, Transcript


def make_transcript():
    return Transcript("fr", 0.98, 3725.5, "whisper", "large-v3", [
        Segment(0.0, 2.5, " Bonjour à tous. "),
        Segment(3661.2, 3665.04, "Fin de l'enregistrement.", language="fr"),
    ])


def test_srt_and_vtt_timestamps():
    t = make_transcript()
    srt = formats.to_srt(t)
    assert "1\n00:00:00,000 --> 00:00:02,500\nBonjour à tous." in srt
    assert "2\n01:01:01,200 --> 01:01:05,040\nFin de l'enregistrement." in srt
    vtt = formats.to_vtt(t)
    assert vtt.startswith("WEBVTT")
    assert "01:01:01.200 --> 01:01:05.040" in vtt
    assert formats.to_txt(t) == "Bonjour à tous.\nFin de l'enregistrement.\n"
    assert '"language": "fr"' in formats.to_json(t)


def test_plan_chunks_prefers_silences():
    chunks = audio.plan_chunks(1000, [(230, 232), (500, 501)], target=240)
    assert chunks[0] == (0.0, 231.0)
    # Contiguous and covering the whole file
    assert chunks[-1][1] == 1000
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    assert all(e - s <= 240 * 1.25 + 1e-6 for s, e in chunks)


def test_plan_chunks_three_hours_without_silence():
    chunks = audio.plan_chunks(3 * 3600, [], target=240)
    assert len(chunks) == 45
    assert chunks[-1] == (10560.0, 10800)


def test_plan_chunks_short_file():
    assert audio.plan_chunks(100, [], target=240) == [(0.0, 100)]


def test_split_timed():
    parts = split_timed("Première phrase. Deuxième phrase plus longue !", 10, 20)
    assert [p[2] for p in parts] == ["Première phrase.", "Deuxième phrase plus longue !"]
    assert parts[0][0] == 10 and abs(parts[-1][1] - 20) < 1e-9


def test_languages_share():
    t = make_transcript()
    t.segments.append(Segment(4000, 4007.5, "Hello", language="en"))
    assert list(t.languages) == ["en", "fr"]


@pytest.fixture
def tone_with_gap(tmp_path) -> Path:
    """2 s de bip, 1.5 s de silence, 2 s de bip — en MP3."""
    out = tmp_path / "tone.mp3"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=1.5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-filter_complex", "[0]aformat=channel_layouts=stereo:sample_rates=44100[a];"
                           "[2]aformat=channel_layouts=stereo:sample_rates=44100[c];"
                           "[a][1][c]concat=n=3:v=0:a=1",
        str(out)], check=True)
    return out


def test_ffmpeg_helpers(tone_with_gap, tmp_path):
    assert abs(audio.probe_duration(tone_with_gap) - 5.5) < 0.15
    wav = audio.to_wav16k(tone_with_gap, tmp_path / "a.wav")
    assert abs(audio.probe_duration(wav) - 5.5) < 0.15
    silences = audio.detect_silences(wav)
    assert any(1.9 < s < 2.3 and 3.3 < e < 3.7 for s, e in silences)
    piece = audio.extract_segment(wav, tmp_path / "p.wav", 1.0, 3.0)
    assert abs(audio.probe_duration(piece) - 2.0) < 0.05


def test_unreadable_file(tmp_path):
    bad = tmp_path / "x.mp3"
    bad.write_bytes(b"not audio")
    with pytest.raises(audio.AudioError):
        audio.probe_duration(bad)
