# stereo-cam

Leave a video call without a fuss. Hold up a hand sign and your webcam picture
dissolves into a photo of your empty room, so you can step away (or close the
laptop) without the others watching you leave.

stereo-cam reads your webcam, watches for the sign with
[MediaPipe](https://ai.google.dev/edge/mediapipe), and sends the result to a
virtual camera. In Teams, Zoom or Meet, pick that virtual camera instead of your
real one.

## How it works

1. **Capture the room.** At startup you get a preview window. Step out of the
   shot, press `c`, and a 3 second countdown takes a photo of the empty room.
2. **Carry on as usual.** Your picture goes to the virtual camera. A preview
   window shows what the camera sees, with a small panel on top.
3. **Make the sign.** Hold both hands up and touch your thumb tips together and
   your index tips together, like a small diamond, for half a second. Your
   picture dissolves into the room photo. (You can pick another sign, see
   [Signs](#signs).)
4. **Come back.** Hold the sign again and the picture dissolves back to you.

The panel shows the frame rate, the processing time per frame, what each hand
is doing with a bar for the hold, and the state (`LIVE`, `DISSOLVING` or `GONE`).
The preview is only for you: the virtual camera gets the clean picture.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- A webcam
- [OBS Studio](https://obsproject.com/) installed, which provides the virtual
  camera that [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) sends to.
  On a Mac you may need to start OBS's *Virtual Camera* once so it is set up,
  and then stop it again, so stereo-cam can use it.
- On macOS, allow camera access for the terminal app you run it from
  (System Settings, Privacy & Security, Camera).

## Setup

```sh
uv sync
curl -o src/stereo_cam/models/gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

The second command downloads the gesture model (about 8 MB). It is not kept in
git.

## Run

```sh
uv run stereo-cam
uv run stereo-cam --config configs/snappy.toml
```

Without `--config` the built-in defaults are used. Then, in your video call,
choose **OBS Virtual Camera** as the camera.

### Keys

| Key | Where | What it does |
|-----|-------|--------------|
| `c` | capture window | Start the countdown that photographs the empty room |
| `r` | preview window | Reset to `LIVE` right away |
| `q` | preview window | Quit |

The keys only work while a window is focused.

## Settings and profiles

Settings live in a TOML file. Every setting is optional: anything you leave out
keeps its default. A typo, an unknown setting or an out-of-range value stops
the app with a message that says what is wrong.

| Setting | Default | Meaning |
|---------|---------|---------|
| `trigger_gesture` | `"FingertipsTouching"` | The sign. `FingertipsTouching`, or one of `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou` |
| `trigger_threshold` | `0.8` | Minimum confidence, from 0 to 1 |
| `hold_duration` | `0.5` | Seconds the sign must be held |
| `hold_grace` | `0.1` | Seconds the sign may drop out without resetting the hold |
| `dissolve_duration` | `1.0` | Seconds the dissolve takes |
| `effect` | `"glitch"` | How you disappear (see below) |
| `requested_fps` | `30` | Frame rate asked of the camera (it may refuse) |
| `camera_index` | `0` | Which camera to use |
| `countdown_seconds` | `3` | Length of the capture countdown |

The profiles in `configs/`:

- `default.toml`: all the defaults, with comments.
- `snappy.toml`: quick to trigger and quick to fade, with the `pixelate` effect.
  More chance of triggering by accident.
- `forgiving.toml`: a longer hold, a more tolerant recognizer and a slow
  `fade`.
- `victory.toml`: triggers on a single-hand Victory sign (the "V") instead of the
  two-hand sign.

Copy one to make your own.

### Signs

`FingertipsTouching` is the default, and it is our own sign: use both hands and
touch your thumb tips together and your index tips together, like a small
diamond. It needs both hands in view. It measures how close the tips are
compared to the size of your hand, so it works at any distance. How close
counts as touching is `TOUCH_THRESHOLD` in `src/stereo_cam/gestures.py`.

The other seven come from MediaPipe's gesture model, and work with a single
hand: `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`,
`Victory` and `ILoveYou`. Use `configs/victory.toml`, or set `trigger_gesture`
in your own file.

### Effects

| Effect | Look |
|--------|------|
| `fade` | A plain cross-fade to the empty room |
| `pixelate` | Small blocks switch to the room one by one, in a random order |
| `glitch` | The colour channels split apart and bands of the picture tear sideways |

To add an effect, write a function that takes `(frame, background, amount)`
and returns a frame (`amount` goes from 0, all you, to 1, all room), and add
it to `EFFECTS` in `src/stereo_cam/effects.py`. It can then be used as the
`effect` setting.

## Tips and limits

- **Step out of the shot when you capture the room.** If you are in the photo,
  you are part of the "empty" room and will not dissolve.
- **Capture in the light you will use.** The photo is taken once. If the sun
  moves or the camera changes its exposure, the room no longer matches and the
  swap will show. Quit with `q` and start again to take a new photo when the
  light changes.
- **Single-hand signs work with either hand.** `FingertipsTouching` needs both
  hands in view.
- MediaPipe prints a few lines of its own log at startup. They are harmless.
- Close other programs that use the camera, or stereo-cam cannot open it.

## Project layout

```
configs/            example settings files
src/stereo_cam/
  main.py           startup, background capture and the main loop
  config.py         the Config settings and the TOML loader
  effects.py        the dissolve effects and the EFFECTS registry
  drawing.py        the preview panel and the text and hand drawing
  gestures.py       reading MediaPipe's result and checking for the sign
  hold_timer.py     how long a sign has been held
  frame_reader.py   reading the camera and surviving hiccups
  models/           the downloaded gesture model
```

## Development

The code is formatted and checked with [ruff](https://docs.astral.sh/ruff/):

```sh
uv run ruff format
uv run ruff check
```
