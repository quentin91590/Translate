import subprocess
import threading

import pytest
from fastapi.testclient import TestClient

from transcriber.backends.base import Backend, Cancelled
from transcriber.config import Settings
from transcriber.models import Segment, Transcript
from transcriber.server import create_app


class FakeBackend(Backend):
    name = "whisper"
    model_name = "fake"

    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()

    def transcribe(self, wav_path, duration, language, progress, is_cancelled, multilingual=False):
        assert wav_path.exists()
        progress(0.5, "à mi-chemin")
        self.gate.wait(5)
        if is_cancelled():
            raise Cancelled()
        return Transcript(language or "fr", 0.9, duration, self.name, self.model_name,
                          [Segment(0, duration, "Bonjour le monde.")])


@pytest.fixture
def setup(tmp_path):
    backend = FakeBackend()
    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings, backend_factory=lambda name, s: backend)
    audio_file = tmp_path / "in.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=1", str(audio_file)], check=True)
    return TestClient(app), app.state.manager, backend, audio_file


def upload(client, path, **data):
    with open(path, "rb") as f:
        return client.post("/api/jobs", files={"file": ("réunion.wav", f, "audio/wav")}, data=data)


def test_full_flow(setup):
    client, manager, _, audio_file = setup
    r = upload(client, audio_file)
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    manager.wait_idle()
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job
    assert job["detected_language"] == "fr"
    assert abs(job["duration"] - 1.0) < 0.1
    txt = client.get(f"/api/jobs/{job_id}/transcript.txt")
    assert txt.text == "Bonjour le monde.\n"
    assert "réunion.txt" in txt.headers["content-disposition"] or "r%C3%A9union.txt" in txt.headers["content-disposition"]
    assert client.get(f"/api/jobs/{job_id}/transcript.srt").text.startswith("1\n00:00:00,000 -->")
    assert len(client.get("/api/jobs").json()) == 1
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_forced_language_and_bad_backend(setup):
    client, manager, _, audio_file = setup
    job_id = upload(client, audio_file, language="EN").json()["id"]
    manager.wait_idle()
    assert client.get(f"/api/jobs/{job_id}").json()["detected_language"] == "en"
    assert upload(client, audio_file, backend="nope").status_code == 400


def test_cancel_running_job(setup):
    client, manager, backend, audio_file = setup
    backend.gate.clear()
    job_id = upload(client, audio_file).json()["id"]
    for _ in range(100):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "running":
            break
        threading.Event().wait(0.05)
    client.post(f"/api/jobs/{job_id}/cancel")
    backend.gate.set()
    manager.wait_idle()
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    assert client.get(f"/api/jobs/{job_id}/transcript.txt").status_code == 409


def test_invalid_audio_reports_error(setup, tmp_path):
    client, manager, _, _ = setup
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"garbage")
    job_id = upload(client, bad).json()["id"]
    manager.wait_idle()
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "error" and job["error"]


def test_index_and_config(setup):
    client, *_ = setup
    assert "Transcripteur audio" in client.get("/").text
    cfg = client.get("/api/config").json()
    assert set(cfg["backends"]) == {"whisper", "qwen"}
