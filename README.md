# Urban Sound Monitor

A Raspberry Pi project that listens to a street with a microphone, saves only the loud events,
classifies each one (cars, scooters, trucks, horns, sirens, voices, machinery and more), and shows
the results in a web interface you can open from any device on the same network.

It answers questions such as: *How loud is the street, and when? Which sounds contribute most to
the noise? How many horn beeps happen in the morning, and how many different horns (cars) are
honking?*

The whole application is one Python file, `noise_events.py`.

> **Status:** working prototype. The code was developed and tested with synthetic audio and a
> simulated browser; it has not been validated against long-term real-street recordings. Treat
> the numbers as relative indicators until you have calibrated the microphone and checked the
> labels yourself.

---

## Features

**Recording**
- Continuous listening in ~1 s blocks with A-weighted level measurement.
- Adaptive background level, so the threshold follows the street's quiet periods.
- Only events are saved: a WAV file (16 kHz mono) and a spectrogram image per event, in
  `recorded_sound/`, named by start time.
- An event starts when the level rises 8 dB above background and ends after 10 s of quiet
  (repeated honks stay one event). Events longer than 60 s are split.
- An event is also split when a different dominant sound lasts at least 3 s (for example,
  a truck followed by voices).

**Classification**
- [YAMNet](https://tfhub.dev/google/yamnet/1) (521 AudioSet classes), run on the loud blocks only.
- Each event gets one **category** (its dominant sound) plus **other sounds**: any other class
  that reaches 25% in at least one block, with the seconds it was present.
- Signal detectors run on each saved recording and add labels that YAMNet cannot give:
  - horn beeps (count, total horn seconds, distinct pitches as an estimate of distinct horns),
  - sirens (sweeping tones),
  - regular beepers (reversing alarms, evenly spaced beeps),
  - low rumble (engine energy below 200 Hz).
- **Car vs scooter learner:** YAMNet alone is weak at separating small scooters from cars. Label
  at least 8 events of each type by hand, press *Learn from my corrections*, and the model
  relabels your uncorrected events. Cross-validated accuracy is shown.

**Horn fixing**
- Fix any event's beeps in a window with the spectrogram: keep or remove detections, add missed
  beeps by dragging on the spectrogram or typing times, zoom to 64×, and group beeps by horn
  (same pitch = same letter).
- The Horns window's *Learn from fixes* learns two things: a beep filter (applied only at 70%+
  cross-validated accuracy) and the pitch distance that counts as one horn. Other events are
  then re-analysed in the background.
- Your fixes always win over automatic analysis, and they update the event's labels, the
  Horns list and the statistics.

**Web interface** (open `http://<pi-address>:8080`)
- **Events:** one day at a time with day navigation and a date picker. Each row shows time,
  length, average and peak level, a volume square (average left, peak right), category, confidence,
  beeps and horn seconds, other sounds, and a spectrogram with a player.
  - Click a category pill to filter; a ✕ resets filters.
  - Rename categories everywhere, merge them, reset a label to automatic, or confirm one by hand.
  - Double-click an *other sound* pill to make it the category.
  - Sort by any column; column names stay visible while scrolling.
  - Per-category colours, saved to disk.
- **Analysis:**
  - *Recent hours:* events per 15 min, hour, or calendar day, over the last hours or all days.
  - *By hour of day:* the daily pattern (for example, cars in the morning, speech in the evening),
    with weekday filters and an Israeli week option (Sun–Thu, Fri–Sat, or Mon–Sun).
  - *Sources:* which sounds carry the most noise energy (bubble chart or share bars), time above
    a threshold (40–90 dB), and level statistics L10 / L50 / L90 / Leq / max.
  - *Heat map:* weekday × hour (or 15 min) for average level, peak level, minutes of sound per
    day, or events per day.
  - Filters by time of day (day, evening, night), and a free view with every event as a point.
- **Horns:** horn statistics (single versus repeated events, beeps per event, pitch counts, horn
  seconds) with a timeline per event.
- **ℹ How it works:** the live settings, the detection rules, time limits, storage use, blind
  spots, and learning status.
- Light, dark and automatic theme; CSV export of all events with labels and horn data.

**Muted mode**
`--muted` runs only the web interface on the saved data: no microphone, no model. You can still
review, relabel, fix horns, learn and delete recordings.

---

## Hardware

- Raspberry Pi 4 (or newer) with Raspberry Pi OS.
- A microphone. Options, from cheapest to best:
  - USB sound adapter with a mic input (CM108/CM109 chip) + electret mic. Check that the adapter
    has no auto gain (it usually doesn't).
  - Calibrated measurement mic (miniDSP UMIK-1 or Dayton iMM-6 class), with a USB adapter if
    it has a 3.5 mm plug.
  - Cheap all-in-one USB mics are discouraged: automatic gain and noise suppression distort
    distant sounds.
- Outdoors: foam windscreen and a rain cover for the microphone.

### Calibration (phone method)
1. Set a sound-level app on your phone to dB(A) and hold it next to the microphone.
2. Play steady noise (white noise or a fan) at a few levels, and record the readings.
3. The offset is *phone dB(A) minus the script's dBFS(A) reading*. Average it over several
   levels and pass it with `--offset`.

Phone calibration gives roughly ±2–3 dB. It is good for comparing places and hours; it is not a
legal noise measurement.

---

## Installation

```bash
sudo apt install -y libportaudio2 python3-numpy python3-scipy python3-flask
pip install sounddevice --break-system-packages
# TensorFlow Lite runtime (one of these; the script tries them in this order):
pip install tflite-runtime --break-system-packages        # Python 3.11 and older
pip install ai-edge-litert --break-system-packages         # newer Python (e.g. 3.13)
```

If you see `numpy.core.multiarray failed to import`, run
`pip install "numpy<2" --break-system-packages`.

On the first run the script downloads `yamnet.tflite` and `yamnet_class_map.csv` next to itself.
If the download fails, get the TFLite YAMNet classification model and its class map manually
and save them with those exact names.

Find the microphone's device index:

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

Use the index of your input device with `--device`. The default is 4; change it to yours.

---

## Usage

```bash
# record and serve the web interface
python3 noise_events.py --device 4 --offset 94

# web interface only, on saved data (no microphone, no model)
python3 noise_events.py --muted

# re-analyse saved recordings (fills labels, spectrogram strip and vehicle data), then exit
python3 noise_events.py --reprocess            # only events without data
python3 noise_events.py --reprocess all        # every event
```

Then open `http://<pi-address>:8080` from any device on the network. The address is printed at
startup.

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--device N` | 4 | Input device index |
| `--offset dB` | 94.0 | dBFS(A) → dB(A) calibration offset |
| `--delta dB` | 8.0 | Level above background that starts an event |
| `--hold s` | 10.0 | Quiet time that ends an event |
| `--max-len s` | 60.0 | Longest event before it is split |
| `--split s` | 3.0 | Split when a different dominant sound lasts this long (0 = off) |
| `--port N` | 8080 | Web port |
| `--web-only`, `--muted` | off | Web interface only |
| `--reprocess [missing\|all]` | — | Re-analyse saved recordings and exit |

### Companion scripts
- `mic_test.py`: lists devices, records a few seconds to a WAV file, prints peak and RMS levels
  and warns about silence or clipping. Use it to check the microphone before the main script.
- `noise_monitor.py`: an earlier, simpler prototype (level and classification only). It is kept
  for reference; `noise_events.py` replaces it.

---

## Files and data

Created next to `noise_events.py`:

| File | Contents |
|---|---|
| `recorded_sound/*.wav` | One WAV per saved event (16 kHz mono, 16-bit) |
| `recorded_sound/*.png` | Spectrogram of each event (created automatically, also on demand) |
| `events.db` | SQLite: `events` (times, levels, categories, labels, timelines, features), `horn` (automatic beep analysis), `horn_manual` (your beep fixes) |
| `category_colors.json` | Colour per category |
| `category_aliases.json` | Renamed built-in categories, applied to new events too |
| `vehicle_model.json` | Learned car-vs-scooter model |
| `horn_model.json` | Learned beep filter and pitch rule |
| `yamnet.tflite`, `yamnet_class_map.csv` | Model files, downloaded on first run |

Deleting an event in the interface deletes its WAV, spectrogram and database rows.

### Categories
Built-in categories are groups of YAMNet classes, defined in the `CATEGORIES` dictionary at the top
of the script: `voices`, `car`, `scooter_moto`, `car_horn`, `siren`, `truck_bus`, `idling_hum`,
`machinery`, `clatter_impact`, `animals`, `aircraft`, `wind_weather`, plus `other` and
`false_alarm`. Edit the dictionary to add or regroup classes. Categories you create in the
interface (for example `garbage_truck`) are not learned by YAMNet; they are assigned by hand.

---

## Web API (for reference)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/events?day=YYYY-MM-DD` | Events of one day (`before=`, `after=`, `limit=` for paging) |
| POST / DELETE | `/api/events/<id>` | Set category, reset label, or delete event |
| GET | `/api/days` | Days that have events, newest first |
| GET | `/api/stats` | Aggregates for recent, hour-of-day and 15-minute views |
| GET | `/api/sources` | Per-category energy, time, thresholds and level statistics |
| GET | `/api/heat` | Weekday × time grid |
| GET | `/api/horns` | Horn results for a category |
| GET / POST / DELETE | `/api/horn/<id>` | Read, save or reset beep fixes of one event |
| POST | `/api/horn/learn` | Learn from beep fixes |
| POST | `/api/train` | Learn car vs scooter from labels |
| GET / POST | `/api/colors` | Category colours |
| GET / POST | `/api/categories`, `/api/categories/rename` | Category list and renaming |
| GET | `/spec/<file>.png` | Spectrogram; add `t0`, `t1`, `w` for a zoomed window |
| GET | `/audio/<file>` | Recording |
| GET | `/export.csv` | All events with labels and horn data |
| GET | `/api/info` | Settings, storage use, learning status |

---

## Security and privacy

- **The web interface has no login.** It listens on all network interfaces. Run it only on a
  trusted home or office network, and do not expose port 8080 to the internet.
- **Recordings may contain speech.** Loud events are saved as audio. Check your local rules on
  recording people in public space, and delete recordings you don't need.
- Only events are stored, not continuous audio.

---

## Limitations

- **Calibration is approximate** (about ±2–3 dB with a phone). Use a calibrated measurement
  microphone for anything beyond comparisons.
- **Classification is probabilistic.** YAMNet was trained on general audio; it confuses similar
  sounds and does not know specific vehicles. Check labels, and use the fix tools.
- **Horn counts are estimates.** Beeps are counted from tonal bursts; distinct horns are estimated
  from pitch, so two identical horns can look like one, and a drifting pitch can look like two.
  Overlapping horns are not separated.
- **16 kHz sampling.** Frequencies above 8 kHz are not recorded, by design. This is what the
  model expects.
- **Single microphone.** Direction and separation of simultaneous sources are not available.
- **Day boundaries and hours use the Pi's local clock.** Check the time zone on the Pi.
- **Learned models need labels.** The car/scooter learner and the beep filter need examples of
  each class (8 per type at minimum; 20 beeps with both kinds for the beep filter). Until then they
  stay off or unhelpful.

---

## Development notes

- Single file, Flask server, vanilla JavaScript page embedded as a string, SVG/canvas-free
  charts drawn in SVG by hand.
- Tests so far were ad hoc scripts with synthetic audio (sine horns, chirps, engine-like buzzes)
  and a simulated browser (jsdom) against the live server. Before relying on the results, run the
  recorder on real traffic for a few days, correct labels, and check the numbers.
- Good first contributions: a proper test suite with recorded reference clips; a
  per-sound level model (energy per category over time); authentication for the web interface;
  packaging as a `systemd` service.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.

The YAMNet class labels come from AudioSet, which is licensed under CC BY 4.0. Please keep the
attribution below when you redistribute the project.

## Acknowledgements

- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) (Apache-2.0)
  and [AudioSet](https://research.google.com/audioset/) (Google, CC BY 4.0).
- [BirdNET-Pi](https://github.com/Nachtzuster/BirdNET-Pi) and the
  [SONYC](https://wp.nyu.edu/sonyc/) urban sound project, which informed the approach.
