# 🎤 Audio Feature Guide (Simple + Structured)

This document explains all audio features in a clear, beginner-friendly way and groups them into meaningful categories.

---

# 🧠 Core Audio Concepts (Quick Definitions)

### 🔊 What is F0 (Fundamental Frequency)?
- The **pitch of the voice**
- Measured in Hz or semitones
- Higher F0 → higher pitch (e.g., excited voice)
- Lower F0 → deeper voice (e.g., tired, monotone)

---

### 🔉 Loudness
- How **loud or soft** the sound is
- Related to energy/intensity

---

### 🎼 MFCC (Mel-Frequency Cepstral Coefficients)
- Represents the **timbre (quality) of sound**
- Captures how humans perceive sound
- Used heavily in speech recognition

---

### 📊 Spectral Features
- Describe how energy is distributed across frequencies
- Example: Is the sound “bright” (high freq) or “dark” (low freq)?

---

### 🗣️ Formants (F1, F2, F3)
- Resonant frequencies of the vocal tract
- Define vowel sounds and articulation

---

### 🔁 Jitter
- Small variations in pitch (cycle-to-cycle)
- High jitter → unstable/rough voice

---

### 🔊 Shimmer
- Small variations in loudness
- High shimmer → breathy or shaky voice

---

### 🔇 HNR (Harmonics-to-Noise Ratio)
- Measures how “clean” the voice is
- High HNR → clear voice
- Low HNR → noisy/hoarse voice

---

### ⏱️ Speech Timing Features
- Speaking rate, pauses, etc.
- Capture fluency and rhythm

---

---

# 📂 Feature Categories

---

# 🧾 1. Metadata
Basic identifiers for the recording:
- `participant_id`
- `session_id`
- `task_name`
- `transcription`
- `duration`

---

# 🎯 2. Pitch (F0 Features)
Describe voice pitch behavior.

### Main Stats
- Mean pitch → average voice tone
- Std dev → variability in pitch
- Percentiles → distribution of pitch

### Features:
- Mean, std, percentiles (20, 50, 80)
- Range (20–80)
- Rising/falling slopes → how pitch changes

---

# 🔊 3. Loudness / Intensity
Describe how loud the speech is.

### Features:
- Mean loudness
- Variability
- Percentiles
- Rising/falling loudness
- Peaks per second

### Extra:
- Mean intensity (dB)
- Intensity variation

---

# 🎼 4. Spectral Features
Describe frequency distribution.

### Features:
- Spectral flux → how fast sound changes
- Spectral slope → energy balance (low vs high freq)
- Spectral tilt → overall frequency trend

### Shape Features:
- Spectral gravity → center of energy
- Std dev → spread of frequencies
- Skewness → asymmetry
- Kurtosis → peak shape

---

# 🎧 5. MFCC (Timbre Features)
Describe sound quality and texture.

### Features:
- MFCC1–MFCC4
- Mean + variability
- Also computed for voiced-only regions

---

# 🗣️ 6. Voice Quality Features

### 🔁 Jitter (Pitch Stability)
- Local jitter
- RAP, PPQ5, DDP variants

### 🔊 Shimmer (Amplitude Stability)
- Local shimmer
- APQ3, APQ5, APQ11
- DDA shimmer

### 🔇 Harmonics / Noise
- HNR (mean + std)
- Measures clarity of voice

### 🧠 Cepstral
- Cepstral Peak Prominence (CPP)
- Measures voice periodicity

---

# 🧠 7. Formants (Speech Articulation)

### F1, F2, F3:
- Frequency → vowel position
- Bandwidth → sharpness
- Amplitude → strength

### Features:
- Mean + variability
- Also includes alternate naming:
  - `mean_f1_loc`, `mean_f2_loc`, etc.

---

# 🎨 8. Spectral Ratios / Voice Color
Describe tonal balance of voice.

### Features:
- H1-H2 → breathiness
- H1-A3 → vocal effort
- Alpha ratio → low vs high freq energy
- Hammarberg index → spectral balance

---

# 📉 9. Spectral Slopes
Energy trends across frequency ranges.

### Features:
- 0–500 Hz slope
- 500–1500 Hz slope
- Separate for voiced/unvoiced

---

# 🔇 10. Voiced / Unvoiced Structure

### Features:
- Voiced segments per second
- Mean voiced length
- Mean unvoiced length
- Variability of segments

---

# 🗨️ 11. Speech Timing / Fluency

### Features:
- Speaking rate → syllables per second
- Articulation rate → speaking speed without pauses
- Pause rate → pauses per second
- Mean pause duration
- Phonation ratio → speaking vs silence time

---

# 🎯 12. Perceptual Quality Metrics

### Features:
- STOI → intelligibility
- PESQ → speech quality
- SI-SDR → distortion measure

---

# ⚡ Key Takeaways (For ML)

Most useful feature groups for modeling:

### 🧠 Mental health / speech disorders:
- Pitch (F0)
- Jitter / shimmer
- HNR
- Speech timing

### 😴 Sleep apnea / fatigue:
- MFCCs
- Spectral features
- Voice quality
- Pauses + speaking rate

---

# ✅ Summary

This dataset captures:
- **Pitch (how you sound)**
- **Energy (how loud you are)**
- **Timbre (how your voice feels)**
- **Articulation (how you speak)**
- **Timing (how fast/paused you are)**
- **Quality (how clean your voice is)**

Together, these features give a **complete representation of human speech behavior**.