# Jarvis Music Studio

Jarvis Music Studio is a Flask-based AI music generation web app for MIDI dataset upload, music21 preprocessing, LSTM training, MIDI generation, playback management, history, favorites, and analytics.

## Features

- Flask backend with SQLite persistence
- MIDI upload for `.mid` and `.midi` files
- music21 note and chord extraction
- TensorFlow/Keras LSTM model with dropout, dense layers, early stopping, and checkpoints
- `models/music_model.h5` training output
- Prompt, genre, mood, instrument, and duration based generation
- Graceful fallback composer when no trained model exists
- Generated MIDI downloads
- Optional WAV/MP3 export when FluidSynth, FFmpeg, and `soundfont.sf2` are available
- WaveSurfer.js player for rendered audio
- Library with search, rename, favorite, delete, and download
- Chart.js analytics dashboard
- Dark/light responsive UI

## Folder Structure

```text
jarvis_music_studio/
  app.py
  train_model.py
  database.db              # created automatically
  requirements.txt
  README.md
  dataset/                 # add training MIDI files here
  models/                  # music_model.h5 and note_mapping.json
  generated_music/         # generated MIDI/WAV/MP3 files
  templates/
  static/
  services/
```

## Installation

```bash
cd jarvis_music_studio
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Train the LSTM Model

Add MIDI files into `dataset/`, then run:

```bash
python train_model.py
```

The training script parses MIDI files with music21, extracts notes and chords, builds sequences, and saves:

```text
models/music_model.h5
models/note_mapping.json
```

If the dataset is too small, the script exits with a clear message. For a college demo, use at least several MIDI files from one style so the model has patterns to learn.

## Run the Web App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

You can generate music immediately. If no trained model is available, the app uses a deterministic rule-based composer so the project remains demo-ready. Once `music_model.h5` exists, generation will attempt to use the trained LSTM.

## WAV and MP3 Export

MIDI generation works out of the box. WAV/MP3 export requires:

- FluidSynth installed and available on PATH
- FFmpeg installed and available on PATH
- A General MIDI soundfont saved as `soundfont.sf2` inside `jarvis_music_studio/`

When those tools are not present, the app still saves and downloads MIDI files.

## Notes for Submission

This project demonstrates the complete workflow:

```text
Dataset -> Preprocessing -> Training -> Generation -> MIDI Output -> Playback/Download -> Analytics
```

SQLite is initialized automatically as `database.db`. Generated files are stored in `generated_music/`.
