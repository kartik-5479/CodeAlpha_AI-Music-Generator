import json
import math
import random
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

try:
    from music21 import chord, duration, instrument as m21_instrument, note, stream, tempo
except ImportError:
    chord = duration = m21_instrument = note = stream = tempo = None


GENRE_SCALES = {
    "Classical": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
    "Jazz": ["C4", "D4", "Eb4", "F4", "G4", "A4", "Bb4", "C5"],
    "Rock": ["E3", "G3", "A3", "B3", "D4", "E4", "G4"],
    "Pop": ["C4", "D4", "E4", "G4", "A4", "C5"],
    "EDM": ["C3", "Eb3", "F3", "G3", "Bb3", "C4", "Eb4"],
    "Lo-Fi": ["A3", "B3", "C4", "E4", "G4", "A4", "C5"],
    "Bollywood": ["C4", "Db4", "E4", "F4", "G4", "Ab4", "B4", "C5"],
    "Instrumental": ["D4", "E4", "F#4", "A4", "B4", "D5"],
}

KEYWORDS = {
    "dark": ["A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4"],
    "dream": ["F3", "G3", "A3", "C4", "D4", "E4", "G4", "A4"],
    "epic": ["D3", "F3", "G3", "A3", "C4", "D4", "F4", "A4"],
    "happy": ["C4", "D4", "E4", "G4", "A4", "B4", "C5"],
    "rain": ["A3", "C4", "D4", "E4", "G4", "A4", "C5"],
    "sad": ["D3", "F3", "G3", "A3", "C4", "D4", "F4"],
    "space": ["C3", "D3", "Eb3", "G3", "A3", "C4", "D4"],
}

CHORD_PROGRESSIONS = {
    "Classical": (0, 4, 5, 3),
    "Jazz": (0, 3, 4, 2),
    "Rock": (0, 4, 5, 6),
    "Pop": (0, 4, 5, 3),
    "EDM": (0, 5, 3, 4),
    "Lo-Fi": (0, 5, 3, 4),
    "Bollywood": (0, 4, 1, 5),
    "Instrumental": (0, 3, 4, 5),
}

MOOD_CONFIG = {
    "Happy": {"tempo": 132, "rests": 0.08, "step": 2},
    "Sad": {"tempo": 76, "rests": 0.18, "step": 1},
    "Relaxing": {"tempo": 84, "rests": 0.16, "step": 1},
    "Energetic": {"tempo": 148, "rests": 0.05, "step": 3},
    "Romantic": {"tempo": 92, "rests": 0.12, "step": 1},
    "Focus": {"tempo": 96, "rests": 0.1, "step": 1},
    "Motivational": {"tempo": 124, "rests": 0.06, "step": 2},
    "Cinematic": {"tempo": 68, "rests": 0.2, "step": 2},
}

INSTRUMENT_CLASSES = {
    "Piano": "Piano",
    "Guitar": "AcousticGuitar",
    "Violin": "Violin",
    "Flute": "Flute",
    "Drums": "Woodblock",
    "Bass": "ElectricBass",
    "Synth": "ElectricPiano",
    "Tabla": "Woodblock",
}

PROGRAMS = {
    "Piano": 0,
    "Guitar": 24,
    "Violin": 40,
    "Flute": 73,
    "Drums": 115,
    "Bass": 33,
    "Synth": 88,
    "Tabla": 115,
}


@dataclass
class GenerationRequest:
    prompt: str
    genre: str
    mood: str
    instrument: str
    duration: int


@dataclass
class GenerationResult:
    title: str
    prompt: str
    genre: str
    mood: str
    instrument: str
    duration: int
    midi_filename: str
    wav_filename: str | None
    mp3_filename: str | None
    model_used: bool

    def to_dict(self):
        return asdict(self)


class MusicGenerator:
    def __init__(self, model_path: Path, mapping_path: Path, output_dir: Path, dataset_dir: Path | None = None):
        self.model_path = model_path
        self.mapping_path = mapping_path
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        self.dataset_cache_path = model_path.parent / "dataset_note_cache.json"
        self._dataset_sequences: list[list[int]] | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.duration = max(10, min(180, request.duration))
        title = self._make_title(request)
        basename = f"{uuid4().hex}_{title.lower().replace(' ', '_')[:45]}"
        midi_path = self.output_dir / f"{basename}.mid"

        model_used = False
        sequence = None
        if self.model_path.exists() and self.mapping_path.exists():
            try:
                sequence = self._generate_with_model(request)
                model_used = True
            except Exception as exc:
                print(f"Model generation fallback: {exc}")

        if sequence is None:
            sequence = self._generate_from_dataset(request)

        if sequence is None:
            sequence = self._generate_rule_based(request)

        sequence = self._polish_sequence(sequence, request)
        self._sequence_to_midi(sequence, request, midi_path)
        wav_name = self._sequence_to_wav(sequence, request, midi_path.with_suffix(".wav"))
        mp3_name = self._try_render_mp3(midi_path.with_suffix(".wav"))
        return GenerationResult(
            title=title,
            prompt=request.prompt or "AI generated music",
            genre=request.genre,
            mood=request.mood,
            instrument=request.instrument,
            duration=request.duration,
            midi_filename=midi_path.name,
            wav_filename=wav_name,
            mp3_filename=mp3_name,
            model_used=model_used,
        )

    def _generate_with_model(self, request: GenerationRequest) -> list[str]:
        from tensorflow.keras.models import load_model

        model = load_model(self.model_path)
        note_to_int = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        int_to_note = {number: note_name for note_name, number in note_to_int.items()}
        vocab_size = len(note_to_int)
        pattern = list(np.random.default_rng().integers(0, vocab_size, size=64))
        total_notes = max(24, int(request.duration * 1.8))
        output = []
        for _ in range(total_notes):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(vocab_size)
            prediction = model.predict(prediction_input, verbose=0)[0]
            index = int(np.random.choice(range(vocab_size), p=self._temperature(prediction, 0.9)))
            output.append(int_to_note[index])
            pattern.append(index)
            pattern = pattern[1:]
        return output

    def _generate_rule_based(self, request: GenerationRequest) -> list[str]:
        rng = random.Random(f"{request.prompt}-{request.genre}-{request.mood}-{request.instrument}")
        scale = self._scale_for_request(request)
        config = MOOD_CONFIG.get(request.mood, MOOD_CONFIG["Focus"])
        total_notes = max(32, int(request.duration * 2.2))
        progression = CHORD_PROGRESSIONS.get(request.genre, CHORD_PROGRESSIONS["Lo-Fi"])
        phrase = self._make_motif(rng, scale, config)
        cursor = phrase[0]
        output: list[str] = []
        for i in range(total_notes):
            if rng.random() < config["rests"]:
                output.append("REST")
                continue
            if i % 16 == 0 and len(scale) > 4:
                root = progression[(i // 16) % len(progression)] % max(1, len(scale) - 4)
                output.append(self._triad(scale, root, rng))
                cursor = root + rng.choice([0, 2, 4])
                continue

            motif_value = phrase[i % len(phrase)]
            drift = rng.choice([-1, 0, 0, 1, config["step"]])
            move = motif_value - cursor + drift
            cursor = max(0, min(len(scale) - 1, cursor + move))
            if i % 16 in {14, 15}:
                cursor = progression[((i // 16) + 1) % len(progression)] % len(scale)
            output.append(scale[cursor])
        return output

    def _generate_from_dataset(self, request: GenerationRequest) -> list[str] | None:
        sequences = self._load_dataset_sequences()
        usable = [sequence for sequence in sequences if len(sequence) >= 16]
        if not usable:
            return None

        rng = random.Random(f"dataset-{request.prompt}-{request.genre}-{request.mood}-{request.instrument}")
        source = rng.choice(usable)
        total_notes = max(32, int(request.duration * 2.2))
        start = rng.randrange(0, max(1, len(source) - total_notes))
        window = source[start : start + total_notes]
        if len(window) < total_notes:
            window = (window * ((total_notes // max(1, len(window))) + 1))[:total_notes]

        output: list[str] = []
        for index, midi_note in enumerate(window):
            midi_note = self._fit_midi_to_scale(midi_note, request)
            if index % 12 == 0 and index + 4 < len(window):
                chord_notes = [
                    midi_note,
                    self._fit_midi_to_scale(window[index + 2], request),
                    self._fit_midi_to_scale(window[index + 4], request),
                ]
                output.append(".".join(self._midi_to_pitch_name(note_number) for note_number in chord_notes))
            else:
                output.append(self._midi_to_pitch_name(midi_note))
        return output

    def _polish_sequence(self, sequence: list[str], request: GenerationRequest) -> list[str]:
        rng = random.Random(f"polish-{request.prompt}-{request.genre}-{request.mood}-{request.instrument}")
        scale = self._scale_for_request(request)
        config = MOOD_CONFIG.get(request.mood, MOOD_CONFIG["Focus"])
        progression = CHORD_PROGRESSIONS.get(request.genre, CHORD_PROGRESSIONS["Lo-Fi"])
        target = max(36, int(request.duration * 2.4))
        source = sequence or self._generate_rule_based(request)
        polished: list[str] = []

        for index in range(target):
            if index % 32 == 0 and index > 0:
                polished.extend(["REST", self._triad(scale, progression[(index // 32) % len(progression)], rng)])
            if index % 16 == 0:
                root = progression[(index // 16) % len(progression)] % max(1, len(scale) - 4)
                polished.append(self._triad(scale, root, rng))
                continue

            item = source[index % len(source)]
            if item == "REST":
                polished.append(item)
                continue
            if rng.random() < config["rests"] * 0.35:
                polished.append("REST")
                continue
            if "." in item and rng.random() < 0.45:
                polished.append(item)
                continue

            midi_notes = self._item_to_midi_notes(item)
            note_number = self._fit_midi_to_scale(midi_notes[0] + rng.choice([-12, 0, 0, 0, 12]), request)
            if request.instrument in {"Bass"}:
                note_number -= 12
            polished.append(self._midi_to_pitch_name(note_number))

            if request.mood in {"Energetic", "Happy", "Motivational"} and index % 8 in {3, 7} and rng.random() < 0.6:
                neighbor = self._fit_midi_to_scale(note_number + rng.choice([-2, 2, 4]), request)
                polished.append(self._midi_to_pitch_name(neighbor))

        return polished[: max(target, 1)]

    def _load_dataset_sequences(self) -> list[list[int]]:
        if self._dataset_sequences is not None:
            return self._dataset_sequences
        if self.dataset_cache_path.exists():
            try:
                cached = json.loads(self.dataset_cache_path.read_text(encoding="utf-8"))
                self._dataset_sequences = [sequence for sequence in cached if isinstance(sequence, list)]
                return self._dataset_sequences
            except (OSError, json.JSONDecodeError):
                pass
        self._dataset_sequences = self._build_dataset_cache()
        return self._dataset_sequences

    def refresh_dataset_cache(self) -> int:
        self._dataset_sequences = self._build_dataset_cache()
        return len(self._dataset_sequences)

    def _build_dataset_cache(self) -> list[list[int]]:
        if self.dataset_dir is None or not self.dataset_dir.exists():
            return []
        midi_files = sorted(
            list(self.dataset_dir.rglob("*.mid")) + list(self.dataset_dir.rglob("*.midi")),
            key=lambda path: path.stat().st_size if path.exists() else 0,
        )
        rng = random.Random("jarvis-dataset-cache")
        if len(midi_files) > 600:
            midi_files = rng.sample(midi_files, 600)

        sequences: list[list[int]] = []
        for midi_file in midi_files:
            notes = self._read_midi_note_numbers(midi_file)
            if len(notes) >= 16:
                sequences.append(notes[:512])
            if len(sequences) >= 240:
                break

        self.dataset_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_cache_path.write_text(json.dumps(sequences), encoding="utf-8")
        return sequences

    def _read_midi_note_numbers(self, midi_file: Path) -> list[int]:
        try:
            data = midi_file.read_bytes()
        except OSError:
            return []
        notes: list[int] = []
        index = 0
        while True:
            track_start = data.find(b"MTrk", index)
            if track_start < 0 or track_start + 8 > len(data):
                break
            length = int.from_bytes(data[track_start + 4 : track_start + 8], "big")
            track = data[track_start + 8 : track_start + 8 + length]
            notes.extend(self._read_midi_track_notes(track))
            index = track_start + 8 + length
        return [note_number for note_number in notes if 24 <= note_number <= 108]

    def _read_midi_track_notes(self, track: bytes) -> list[int]:
        notes: list[int] = []
        index = 0
        running_status = None
        while index < len(track):
            _, index = self._read_var_len(track, index)
            if index >= len(track):
                break
            status = track[index]
            if status & 0x80:
                index += 1
                if status == 0xFF:
                    if index >= len(track):
                        break
                    index += 1
                    length, index = self._read_var_len(track, index)
                    index += length
                    continue
                if status in {0xF0, 0xF7}:
                    length, index = self._read_var_len(track, index)
                    index += length
                    continue
                running_status = status
            elif running_status is not None:
                status = running_status
            else:
                break

            event_type = status & 0xF0
            data_len = 1 if event_type in {0xC0, 0xD0} else 2
            if index + data_len > len(track):
                break
            first = track[index]
            second = track[index + 1] if data_len == 2 else 0
            index += data_len
            if event_type == 0x90 and second > 0:
                notes.append(first)
        return notes

    @staticmethod
    def _read_var_len(data: bytes, index: int) -> tuple[int, int]:
        value = 0
        while index < len(data):
            byte = data[index]
            index += 1
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                break
        return value, index

    def _sequence_to_midi(self, sequence: list[str], request: GenerationRequest, midi_path: Path) -> None:
        if stream is None:
            self._sequence_to_midi_bytes(sequence, request, midi_path)
            return

        score = stream.Stream()
        inst_name = INSTRUMENT_CLASSES.get(request.instrument, "Piano")
        inst_class = getattr(m21_instrument, inst_name, m21_instrument.Piano)
        score.append(inst_class())
        score.append(tempo.MetronomeMark(number=MOOD_CONFIG.get(request.mood, MOOD_CONFIG["Focus"])["tempo"]))
        offset = 0.0
        rng = random.Random(f"midi-{request.prompt}-{request.genre}-{request.mood}-{request.instrument}")
        for index, item in enumerate(sequence):
            ql = self._note_length(request, index, rng)
            if item == "REST":
                offset += ql
                continue
            if "." in item:
                pitches = item.split(".")
                if all(part.isdigit() for part in pitches):
                    notes = [note.Note(int(part)) for part in pitches]
                else:
                    notes = [note.Note(part) for part in pitches]
                new_chord = chord.Chord(notes)
                new_chord.duration = duration.Duration(ql * 1.5)
                new_chord.volume.velocity = self._velocity(request, index, chord=True)
                score.insert(offset, new_chord)
            else:
                new_note = note.Note(item)
                new_note.duration = duration.Duration(ql)
                new_note.volume.velocity = self._velocity(request, index)
                score.insert(offset, new_note)
            offset += ql * random.Random(f"{index}-{request.prompt}").uniform(0.94, 1.04)
            if offset >= max(8, request.duration * 2):
                break
        score.write("midi", fp=str(midi_path))

    def _sequence_to_midi_bytes(self, sequence: list[str], request: GenerationRequest, midi_path: Path) -> None:
        ticks_per_beat = 480
        bpm = MOOD_CONFIG.get(request.mood, MOOD_CONFIG["Focus"])["tempo"]
        tempo_us = int(60_000_000 / bpm)
        channel = 9 if request.instrument in {"Drums", "Tabla"} else 0
        program = PROGRAMS.get(request.instrument, 0)
        events: list[tuple[int, bytes]] = [
            (0, bytes([0xC0 | channel, program])),
        ]
        current_tick = 0
        rng = random.Random(f"midi-bytes-{request.prompt}-{request.genre}-{request.mood}-{request.instrument}")

        for index, item in enumerate(sequence):
            if current_tick >= max(8, request.duration * 2) * ticks_per_beat:
                break
            note_ticks = int(ticks_per_beat * self._note_length(request, index, rng))
            if item == "REST":
                current_tick += note_ticks
                continue
            midi_notes = self._item_to_midi_notes(item)
            velocity = self._velocity(request, index, chord="." in item)
            for midi_note in midi_notes:
                events.append((current_tick, bytes([0x90 | channel, midi_note, velocity])))
            length = int(note_ticks * (1.5 if "." in item else 1))
            for midi_note in midi_notes:
                events.append((current_tick + length, bytes([0x80 | channel, midi_note, 0])))
            current_tick += note_ticks

        events.sort(key=lambda event: event[0])
        track = bytearray()
        track.extend(b"\x00\xff\x51\x03" + tempo_us.to_bytes(3, "big"))
        last_tick = 0
        for tick, payload in events:
            track.extend(self._var_len(tick - last_tick))
            track.extend(payload)
            last_tick = tick
        track.extend(b"\x00\xff\x2f\x00")

        header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + ticks_per_beat.to_bytes(2, "big")
        midi_path.write_bytes(header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track))

    def _item_to_midi_notes(self, item: str) -> list[int]:
        if "." in item:
            parts = item.split(".")
            if all(part.isdigit() for part in parts):
                return [60 + int(part) for part in parts]
            return [self._pitch_to_midi(part) for part in parts]
        return [self._pitch_to_midi(item)]

    @staticmethod
    def _pitch_to_midi(pitch: str) -> int:
        names = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
        pitch = pitch.strip()
        if pitch.isdigit():
            return max(0, min(127, 60 + int(pitch)))
        octave = int(pitch[-1]) if pitch[-1].isdigit() else 4
        name = pitch[:-1] if pitch[-1].isdigit() else pitch
        return max(0, min(127, 12 * (octave + 1) + names.get(name, 0)))

    @staticmethod
    def _var_len(value: int) -> bytes:
        value = max(0, value)
        buffer = [value & 0x7F]
        value >>= 7
        while value:
            buffer.insert(0, (value & 0x7F) | 0x80)
            value >>= 7
        return bytes(buffer)

    def _try_render_audio(self, midi_path: Path) -> tuple[str | None, str | None]:
        # Optional export: if FluidSynth is installed and a soundfont is configured, create WAV/MP3.
        soundfont = self.output_dir.parent / "soundfont.sf2"
        fluidsynth = shutil.which("fluidsynth")
        ffmpeg = shutil.which("ffmpeg")
        if not fluidsynth or not soundfont.exists():
            return None, None
        wav_path = midi_path.with_suffix(".wav")
        subprocess.run([fluidsynth, "-ni", str(soundfont), str(midi_path), "-F", str(wav_path), "-r", "44100"], check=False)
        mp3_name = None
        if ffmpeg and wav_path.exists():
            mp3_path = midi_path.with_suffix(".mp3")
            subprocess.run([ffmpeg, "-y", "-i", str(wav_path), str(mp3_path)], check=False)
            mp3_name = mp3_path.name if mp3_path.exists() else None
        return (wav_path.name if wav_path.exists() else None, mp3_name)

    def _sequence_to_wav(self, sequence: list[str], request: GenerationRequest, wav_path: Path) -> str | None:
        sample_rate = 44100
        bpm = MOOD_CONFIG.get(request.mood, MOOD_CONFIG["Focus"])["tempo"]
        beat_seconds = 60.0 / bpm
        rng = random.Random(f"wav-{request.prompt}-{request.genre}-{request.mood}-{request.instrument}")
        max_seconds = max(8, request.duration)
        samples: list[float] = []

        for index, item in enumerate(sequence):
            if len(samples) >= int(max_seconds * sample_rate):
                break
            note_seconds = beat_seconds * self._note_length(request, index, rng)
            duration_seconds = note_seconds * (1.5 if "." in item else 1.0)
            count = max(1, int(duration_seconds * sample_rate))
            if item == "REST":
                samples.extend([0.0] * count)
                continue
            freqs = [self._midi_to_frequency(midi_note) for midi_note in self._item_to_midi_notes(item)]
            samples.extend(self._synthesize_note(freqs, count, sample_rate, request.instrument))

        if not samples:
            return None

        samples = self._master_samples(samples, sample_rate)
        peak = max(max(abs(sample) for sample in samples), 0.01)
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            frames = bytearray()
            for sample in samples:
                value = int(max(-1.0, min(1.0, sample / peak * 0.86)) * 32767)
                frames.extend(value.to_bytes(2, "little", signed=True))
            wav.writeframes(bytes(frames))
        return wav_path.name

    def _synthesize_note(self, frequencies: list[float], count: int, sample_rate: int, instrument: str) -> list[float]:
        attack = max(1, int(0.018 * sample_rate))
        release = max(1, int(0.08 * sample_rate))
        sustain = 0.58 if instrument in {"Piano", "Guitar", "Tabla", "Drums"} else 0.75
        output: list[float] = []

        for index in range(count):
            t = index / sample_rate
            if index < attack:
                envelope = index / attack
            elif index > count - release:
                envelope = max(0.0, (count - index) / release)
            else:
                envelope = sustain

            value = 0.0
            for frequency in frequencies:
                if instrument in {"Drums", "Tabla"}:
                    value += math.sin(2 * math.pi * 90 * t) * math.exp(-t * 9)
                    value += random.uniform(-0.35, 0.35) * math.exp(-t * 18)
                elif instrument in {"Guitar", "Bass"}:
                    value += math.sin(2 * math.pi * frequency * t)
                    value += 0.35 * math.sin(2 * math.pi * frequency * 2 * t)
                    value += 0.15 * math.sin(2 * math.pi * frequency * 3 * t)
                elif instrument in {"Synth"}:
                    value += math.sin(2 * math.pi * frequency * t)
                    value += 0.45 * math.sin(2 * math.pi * (frequency * 1.005) * t)
                else:
                    value += math.sin(2 * math.pi * frequency * t)
                    value += 0.22 * math.sin(2 * math.pi * frequency * 2 * t)
            output.append((value / max(1, len(frequencies))) * envelope * 0.42)
        return output

    def _scale_for_request(self, request: GenerationRequest) -> list[str]:
        prompt = request.prompt.lower()
        for keyword, scale in KEYWORDS.items():
            if keyword in prompt:
                return scale
        if request.mood == "Sad":
            return KEYWORDS["sad"]
        if request.mood == "Cinematic":
            return KEYWORDS["epic"]
        if request.mood == "Happy":
            return KEYWORDS["happy"]
        return GENRE_SCALES.get(request.genre, GENRE_SCALES["Lo-Fi"])

    @staticmethod
    def _make_motif(rng: random.Random, scale: list[str], config: dict[str, float | int]) -> list[int]:
        start = rng.randrange(max(1, len(scale)))
        motif = [start]
        for _ in range(7):
            step = int(config["step"])
            motif.append(max(0, min(len(scale) - 1, motif[-1] + rng.choice([-step, -1, 0, 1, step]))))
        return motif

    @staticmethod
    def _triad(scale: list[str], root: int, rng: random.Random) -> str:
        root = root % max(1, len(scale) - 4)
        tones = [scale[root], scale[root + 2], scale[root + 4]]
        if rng.random() < 0.28 and root + 6 < len(scale):
            tones.append(scale[root + 6])
        return ".".join(tones)

    def _fit_midi_to_scale(self, midi_note: int, request: GenerationRequest) -> int:
        scale_midis = [self._pitch_to_midi(pitch) % 12 for pitch in self._scale_for_request(request)]
        octave = max(2, min(6, int(midi_note) // 12 - 1))
        candidates = [12 * (octave + 1) + pitch_class for pitch_class in scale_midis]
        best = min(candidates, key=lambda candidate: abs(candidate - midi_note))
        return max(36, min(96, best))

    @staticmethod
    def _note_length(request: GenerationRequest, index: int, rng: random.Random) -> float:
        if request.mood in {"Energetic", "Happy", "Motivational"}:
            base = rng.choice([0.38, 0.5, 0.5, 0.75])
        elif request.mood in {"Relaxing", "Sad", "Cinematic"}:
            base = rng.choice([0.75, 0.75, 1.0, 1.25])
        else:
            base = rng.choice([0.5, 0.75, 0.75, 1.0])
        if index % 16 == 0:
            base *= 1.35
        return base

    @staticmethod
    def _velocity(request: GenerationRequest, index: int, chord: bool = False) -> int:
        base = 92 if request.mood in {"Energetic", "Motivational", "Happy"} else 74
        if request.mood in {"Sad", "Relaxing", "Cinematic"}:
            base -= 8
        accent = 13 if index % 8 == 0 else 5 if index % 4 == 0 else 0
        if chord:
            base -= 6
        return max(42, min(116, base + accent))

    @staticmethod
    def _master_samples(samples: list[float], sample_rate: int) -> list[float]:
        if not samples:
            return samples
        delay = max(1, int(0.115 * sample_rate))
        room = max(1, int(0.037 * sample_rate))
        mastered: list[float] = []
        for index, sample in enumerate(samples):
            wet = sample
            if index >= delay:
                wet += 0.16 * samples[index - delay]
            if index >= room:
                wet += 0.08 * samples[index - room]
            fade_in = min(1.0, index / max(1, int(0.05 * sample_rate)))
            fade_out = min(1.0, (len(samples) - index) / max(1, int(0.08 * sample_rate)))
            mastered.append(math.tanh(wet * 1.25) * fade_in * fade_out)
        return mastered

    @staticmethod
    def _midi_to_frequency(midi_note: int) -> float:
        return 440.0 * (2 ** ((midi_note - 69) / 12))

    @staticmethod
    def _midi_to_pitch_name(midi_note: int) -> str:
        names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        midi_note = max(0, min(127, int(midi_note)))
        octave = (midi_note // 12) - 1
        return f"{names[midi_note % 12]}{octave}"

    def _try_render_mp3(self, wav_path: Path) -> str | None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or not wav_path.exists():
            return None
        mp3_path = wav_path.with_suffix(".mp3")
        subprocess.run([ffmpeg, "-y", "-i", str(wav_path), str(mp3_path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp3_path.name if mp3_path.exists() else None

    def _make_title(self, request: GenerationRequest) -> str:
        seed_words = [word.strip(".,!?").title() for word in request.prompt.split() if len(word) > 3]
        phrase = " ".join(seed_words[:3]) if seed_words else f"{request.mood} {request.genre}"
        return f"{phrase} {request.instrument}".strip()[:80]

    @staticmethod
    def _temperature(predictions, temperature: float):
        predictions = np.asarray(predictions).astype("float64")
        predictions = np.log(np.maximum(predictions, 1e-8)) / temperature
        exp_preds = np.exp(predictions)
        return exp_preds / np.sum(exp_preds)
