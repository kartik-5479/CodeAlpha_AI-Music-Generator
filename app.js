const genres = ["Classical", "Jazz", "Rock", "Pop", "EDM", "Lo-Fi", "Bollywood", "Instrumental"];
const moods = ["Happy", "Sad", "Relaxing", "Energetic", "Romantic", "Focus", "Motivational", "Cinematic"];
const instruments = ["Piano", "Guitar", "Violin", "Flute", "Drums", "Bass", "Synth", "Tabla"];

const state = {
  wavesurfer: null,
  currentSong: null,
  loop: false,
  favoritesOnly: false,
  charts: {}
};

const $ = (selector) => document.querySelector(selector);

function toast(message, type = "info") {
  const node = $("#toast");
  node.textContent = message;
  node.style.borderColor = type === "error" ? "#fb7185" : type === "success" ? "#34d399" : "rgba(255,255,255,.14)";
  node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 3200);
}

function fillSelect(id, values, selected) {
  const select = $(id);
  select.innerHTML = values.map((value) => `<option ${value === selected ? "selected" : ""}>${value}</option>`).join("");
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || "Request failed.");
  return data;
}

function switchView(section) {
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === section));
  document.querySelectorAll(".nav-link").forEach((btn) => btn.classList.toggle("active", btn.dataset.section === section));
  if (section === "analytics") loadAnalytics();
  if (section === "library") loadSongs();
}

function setupWaveSurfer() {
  if (state.wavesurfer) state.wavesurfer.destroy();
  $("#waveform").innerHTML = "";
  if (!window.WaveSurfer) {
    state.wavesurfer = null;
    $("#waveform").innerHTML = '<div class="midi-message">Audio waveform is unavailable. Download generated MIDI files from the links below.</div>';
    return;
  }
  state.wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "#4f46e5",
    progressColor: "#06b6d4",
    cursorColor: "#ffffff",
    height: 140,
    barWidth: 3,
    barRadius: 3,
    normalize: true
  });
  state.wavesurfer.setVolume(Number($("#volume").value));
  state.wavesurfer.on("finish", () => {
    if (state.loop) state.wavesurfer.play(0);
  });
}

function loadTrack(song) {
  song = normalizeSong(song);
  state.currentSong = song;
  $("#currentTitle").textContent = song.title;
  const audioFile = song.wav_file || song.mp3_file;
  const links = [];
  if (song.midi_file) links.push(`<a href="/download/${encodeURIComponent(song.midi_file)}">MIDI</a>`);
  if (song.wav_file) links.push(`<a href="/download/${encodeURIComponent(song.wav_file)}">WAV</a>`);
  if (song.mp3_file) links.push(`<a href="/download/${encodeURIComponent(song.mp3_file)}">MP3</a>`);
  $("#downloadLinks").innerHTML = links.join("");
  if (audioFile) {
    setupWaveSurfer();
    if (state.wavesurfer) {
      state.wavesurfer.load(`/download/${encodeURIComponent(audioFile)}`);
      toast("Loaded audio waveform.");
    }
  } else {
    if (state.wavesurfer) {
      state.wavesurfer.destroy();
      state.wavesurfer = null;
    }
    $("#waveform").innerHTML = '<div class="midi-message">Audio is still being prepared. Generate again or choose a track with a WAV link from the Library.</div>';
    toast("No playable WAV/MP3 file found for this track.", "error");
  }
}

function normalizeSong(song) {
  return {
    ...song,
    midi_file: song.midi_file || song.midi_filename || "",
    wav_file: song.wav_file || song.wav_filename || "",
    mp3_file: song.mp3_file || song.mp3_filename || ""
  };
}

async function generateMusic(event) {
  event.preventDefault();
  const button = event.target.querySelector("button[type=submit]");
  button.disabled = true;
  button.textContent = "Composing...";
  try {
    const payload = {
      prompt: $("#prompt").value,
      genre: $("#genre").value,
      mood: $("#mood").value,
      instrument: $("#instrument").value,
      duration: Number($("#duration").value)
    };
    const data = await api("/api/generate", { method: "POST", body: JSON.stringify(payload) });
    loadTrack(normalizeSong(data.song));
    await loadSongs();
    await loadAnalytics();
    toast(data.song.model_used ? "Generated with trained LSTM model." : "Generated with fallback composer. Train the LSTM for learned style.", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Generate Music";
  }
}

async function loadSongs() {
  const query = new URLSearchParams({
    search: $("#search")?.value || "",
    favorites: state.favoritesOnly ? "1" : "0"
  });
  const data = await api(`/api/songs?${query}`);
  const list = $("#songList");
  if (!data.songs.length) {
    list.innerHTML = '<div class="panel">No generated songs yet.</div>';
    return;
  }
  list.innerHTML = data.songs.map((song) => `
    <article class="song-card">
      <div>
        <h3>${escapeHtml(song.title)} ${song.favorite ? '<span class="favorite">Favorite</span>' : ""}</h3>
        <div class="song-meta">${escapeHtml(song.genre)} / ${escapeHtml(song.mood)} / ${escapeHtml(song.instrument)} / ${song.duration}s / ${escapeHtml(song.created_at || "Unknown date")}</div>
        <div class="song-meta">${escapeHtml(song.prompt)}</div>
      </div>
      <div class="song-actions">
        <button data-action="play" data-id="${song.id}">Play</button>
        <button data-action="favorite" data-id="${song.id}">${song.favorite ? "Unfavorite" : "Favorite"}</button>
        <button data-action="rename" data-id="${song.id}">Rename</button>
        <button data-action="delete" data-id="${song.id}">Delete</button>
        <a href="/download/${encodeURIComponent(song.midi_file)}">MIDI</a>
      </div>
    </article>
  `).join("");
  list.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => handleSongAction(button.dataset.action, Number(button.dataset.id), data.songs));
  });
}

async function handleSongAction(action, id, songs) {
  const song = songs.find((item) => item.id === id);
  if (!song) {
    toast("Song not found.", "error");
    return;
  }
  try {
    if (action === "play") {
      loadTrack(song);
      switchView("studio");
    } else if (action === "favorite") {
      await api(`/api/songs/${id}/favorite`, { method: "POST", body: "{}" });
      await loadSongs();
    } else if (action === "rename") {
      const title = prompt("New title", song.title);
      if (title) {
        await api(`/api/songs/${id}/rename`, { method: "POST", body: JSON.stringify({ title }) });
        await loadSongs();
      }
    } else if (action === "delete" && confirm("Delete this song?")) {
      await api(`/api/songs/${id}`, { method: "DELETE", body: "{}" });
      await loadSongs();
      await loadAnalytics();
    }
  } catch (error) {
    toast(error.message, "error");
  }
}

async function loadAnalytics() {
  const { analytics } = await api("/api/analytics");
  $("#statSongs").textContent = analytics.songs_generated;
  $("#statGenre").textContent = analytics.most_used_genre;
  $("#statMood").textContent = analytics.most_used_mood;
  $("#statListening").textContent = `${analytics.total_listening_time}s`;
  renderChart("genreChart", "Genres", analytics.genres);
  renderChart("moodChart", "Moods", analytics.moods);
}

function renderChart(id, label, rows) {
  const node = $(`#${id}`);
  if (!window.Chart) {
    node.replaceWith(Object.assign(document.createElement("div"), {
      id,
      className: "midi-message",
      textContent: `${label} chart unavailable.`
    }));
    return;
  }
  if (!(node instanceof HTMLCanvasElement)) return;
  if (state.charts[id]) state.charts[id].destroy();
  state.charts[id] = new Chart(node, {
    type: "doughnut",
    data: {
      labels: rows.map((row) => row.label),
      datasets: [{ label, data: rows.map((row) => row.value), backgroundColor: ["#4f46e5", "#7c3aed", "#06b6d4", "#34d399", "#f59e0b", "#fb7185"] }]
    },
    options: { plugins: { legend: { labels: { color: getComputedStyle(document.body).getPropertyValue("--text") } } } }
  });
}

async function uploadDataset(event) {
  event.preventDefault();
  if (!$("#datasetFiles").files.length) {
    toast("Choose at least one MIDI file.", "error");
    return;
  }
  const formData = new FormData();
  Array.from($("#datasetFiles").files).forEach((file) => formData.append("files", file));
  const data = await api("/api/upload-dataset", { method: "POST", body: formData });
  toast(`Uploaded ${data.count} MIDI file(s).`, "success");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

document.addEventListener("DOMContentLoaded", async () => {
  fillSelect("#genre", genres, "Lo-Fi");
  fillSelect("#mood", moods, "Focus");
  fillSelect("#instrument", instruments, "Piano");
  setupWaveSurfer();

  document.querySelectorAll(".nav-link").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.section)));
  $("#generateForm").addEventListener("submit", generateMusic);
  $("#datasetForm").addEventListener("submit", uploadDataset);
  $("#duration").addEventListener("input", () => $("#durationValue").textContent = `${$("#duration").value}s`);
  $("#search").addEventListener("input", loadSongs);
  $("#favoritesOnly").addEventListener("click", () => {
    state.favoritesOnly = !state.favoritesOnly;
    $("#favoritesOnly").classList.toggle("active", state.favoritesOnly);
    loadSongs();
  });
  $("#themeToggle").addEventListener("click", () => {
    document.body.classList.toggle("light");
    $("#themeToggle").textContent = document.body.classList.contains("light") ? "Dark Mode" : "Light Mode";
    loadAnalytics();
  });
  $("#playButton").addEventListener("click", () => state.wavesurfer?.play());
  $("#pauseButton").addEventListener("click", () => state.wavesurfer?.pause());
  $("#stopButton").addEventListener("click", () => state.wavesurfer?.stop());
  $("#loopButton").addEventListener("click", () => {
    state.loop = !state.loop;
    $("#loopButton").classList.toggle("active", state.loop);
  });
  $("#volume").addEventListener("input", () => state.wavesurfer?.setVolume(Number($("#volume").value)));

  await loadSongs();
  await loadAnalytics();
  setTimeout(() => $("#loadingScreen").classList.add("hidden"), 550);
});
