import os
import sys
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.database import (
    close_db,
    delete_song,
    get_analytics,
    get_db,
    init_db,
    insert_song,
    list_songs,
    rename_song,
    toggle_favorite,
    update_listening_time,
)
from services.generator import GENRE_SCALES, INSTRUMENT_CLASSES, MOOD_CONFIG, GenerationRequest, MusicGenerator


DATASET_DIR = BASE_DIR / "dataset"
GENERATED_DIR = BASE_DIR / "generated_music"
MODEL_DIR = BASE_DIR / "models"
ALLOWED_MIDI = {".mid", ".midi"}


def _choice(value: object, allowed: set[str], fallback: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else fallback


def _int_value(value: object, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "jarvis-dev-secret")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    for folder in (DATASET_DIR, GENERATED_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    init_db(BASE_DIR / "database.db")
    app.teardown_appcontext(close_db)

    generator = MusicGenerator(
        model_path=MODEL_DIR / "music_model.h5",
        mapping_path=MODEL_DIR / "note_mapping.json",
        output_dir=GENERATED_DIR,
        dataset_dir=DATASET_DIR,
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/generate", methods=["POST"])
    def generate_music():
        payload = request.get_json(silent=True) or {}
        try:
            generation = generator.generate(
                GenerationRequest(
                    prompt=str(payload.get("prompt", "")).strip(),
                    genre=_choice(payload.get("genre"), set(GENRE_SCALES), "Lo-Fi"),
                    mood=_choice(payload.get("mood"), set(MOOD_CONFIG), "Focus"),
                    instrument=_choice(payload.get("instrument"), set(INSTRUMENT_CLASSES), "Piano"),
                    duration=_int_value(payload.get("duration"), 30),
                )
            )
            song_id = insert_song(
                prompt=generation.prompt,
                genre=generation.genre,
                mood=generation.mood,
                instrument=generation.instrument,
                duration=generation.duration,
                title=generation.title,
                midi_file=generation.midi_filename,
                wav_file=generation.wav_filename,
                mp3_file=generation.mp3_filename,
            )
            song = {
                **generation.to_dict(),
                "id": song_id,
                "midi_file": generation.midi_filename,
                "wav_file": generation.wav_filename,
                "mp3_file": generation.mp3_filename,
            }
            return jsonify({"ok": True, "song": song})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/songs")
    def songs():
        search = request.args.get("search", "")
        favorites = request.args.get("favorites") == "1"
        return jsonify({"ok": True, "songs": list_songs(search=search, favorites=favorites)})

    @app.route("/api/songs/<int:song_id>/favorite", methods=["POST"])
    def favorite(song_id: int):
        try:
            return jsonify({"ok": True, "favorite": toggle_favorite(song_id)})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @app.route("/api/songs/<int:song_id>/rename", methods=["POST"])
    def rename(song_id: int):
        payload = request.get_json(silent=True) or {}
        title = str(payload.get("title", "")).strip()
        if not title:
            return jsonify({"ok": False, "error": "Title is required."}), 400
        exists = get_db().execute("SELECT 1 FROM songs WHERE id = ?", (song_id,)).fetchone()
        if not exists:
            return jsonify({"ok": False, "error": "Song not found."}), 404
        rename_song(song_id, title[:120])
        return jsonify({"ok": True})

    @app.route("/api/songs/<int:song_id>", methods=["DELETE"])
    def remove_song(song_id: int):
        song = get_db().execute("SELECT midi_file, wav_file, mp3_file FROM songs WHERE id = ?", (song_id,)).fetchone()
        if not song:
            return jsonify({"ok": False, "error": "Song not found."}), 404
        delete_song(song_id)
        for key in ("midi_file", "wav_file", "mp3_file"):
            filename = song[key]
            if filename:
                path = GENERATED_DIR / filename
                if path.exists():
                    path.unlink()
        return jsonify({"ok": True})

    @app.route("/api/analytics")
    def analytics():
        return jsonify({"ok": True, "analytics": get_analytics()})

    @app.route("/api/listening", methods=["POST"])
    def listening():
        payload = request.get_json(silent=True) or {}
        update_listening_time(_int_value(payload.get("song_id")), _int_value(payload.get("seconds")))
        return jsonify({"ok": True})

    @app.route("/api/upload-dataset", methods=["POST"])
    def upload_dataset():
        files = request.files.getlist("files")
        saved = []
        for file in files:
            if not file or not file.filename:
                continue
            suffix = Path(file.filename).suffix.lower()
            if suffix not in ALLOWED_MIDI:
                continue
            safe = secure_filename(file.filename)
            target = DATASET_DIR / f"{uuid4().hex}_{safe}"
            file.save(target)
            saved.append(target.name)
        if saved:
            generator.refresh_dataset_cache()
        return jsonify({"ok": True, "saved": saved, "count": len(saved)})

    @app.route("/api/dataset/refresh", methods=["POST"])
    def refresh_dataset():
        count = generator.refresh_dataset_cache()
        return jsonify({"ok": True, "sequences": count})

    @app.route("/download/<path:filename>")
    def download(filename: str):
        safe = secure_filename(filename)
        path = GENERATED_DIR / safe
        if not path.exists():
            return jsonify({"ok": False, "error": "File not found."}), 404
        return send_file(path, as_attachment=True)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=True, host="127.0.0.1", port=port)
