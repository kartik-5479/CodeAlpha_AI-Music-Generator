from pathlib import Path

try:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
except ImportError as exc:
    raise SystemExit("TensorFlow is required for training. Run: pip install -r requirements.txt") from exc

try:
    from services.preprocessing import extract_notes, prepare_sequences, save_mapping
except ImportError as exc:
    raise SystemExit("music21 is required for MIDI preprocessing. Run: pip install -r requirements.txt") from exc


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "music_model.h5"
MAPPING_PATH = MODEL_DIR / "note_mapping.json"


def build_model(sequence_length: int, vocab_size: int) -> Sequential:
    model = Sequential(
        [
            LSTM(256, input_shape=(sequence_length, 1), return_sequences=True),
            Dropout(0.3),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(128),
            BatchNormalization(),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
    return model


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    notes = extract_notes(DATASET_DIR)
    if len(notes) < 80:
        raise SystemExit(
            "Not enough MIDI data found. Add .mid or .midi files to jarvis_music_studio/dataset/ and run again."
        )
    x, y, note_to_int = prepare_sequences(notes, sequence_length=64)
    save_mapping(note_to_int, MAPPING_PATH)
    model = build_model(sequence_length=x.shape[1], vocab_size=y.shape[1])
    checkpoint = ModelCheckpoint(str(MODEL_PATH), monitor="loss", save_best_only=True, mode="min", verbose=1)
    early_stop = EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)
    model.fit(x, y, epochs=80, batch_size=64, callbacks=[checkpoint, early_stop])
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
