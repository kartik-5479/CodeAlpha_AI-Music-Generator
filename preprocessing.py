import json
import os
from pathlib import Path

import numpy as np
from music21 import chord, converter, instrument, note


def extract_notes(dataset_dir: Path) -> list[str]:
    notes: list[str] = []
    max_files = int(os.environ.get("MAX_MIDI_FILES", "1200"))
    midi_files = sorted(
        list(dataset_dir.rglob("*.mid")) + list(dataset_dir.rglob("*.midi")),
        key=lambda path: path.stat().st_size if path.exists() else 0,
    )[:max_files]
    for midi_file in midi_files:
        try:
            parsed = converter.parse(str(midi_file))
            parts = instrument.partitionByInstrument(parsed)
            elements = parts.parts[0].recurse() if parts else parsed.flatten().notes
            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))
        except Exception as exc:
            print(f"Skipped {midi_file.name}: {exc}")
    return notes


def prepare_sequences(notes: list[str], sequence_length: int = 64):
    if len(notes) <= sequence_length:
        raise ValueError("Dataset is too small. Add more MIDI files to dataset/.")
    pitch_names = sorted(set(notes))
    note_to_int = {note_name: number for number, note_name in enumerate(pitch_names)}
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[item] for item in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    x = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(len(pitch_names))
    y = np.eye(len(pitch_names))[network_output]
    return x, y, note_to_int


def save_mapping(note_to_int: dict[str, int], mapping_path: Path) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(note_to_int, indent=2), encoding="utf-8")


def load_mapping(mapping_path: Path) -> dict[str, int]:
    return json.loads(mapping_path.read_text(encoding="utf-8"))
