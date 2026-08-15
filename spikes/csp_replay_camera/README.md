# CSP replay camera spike

**This is an experiment, not a feature.** It exists to answer two questions that
have blocked the same three pieces of work since the project started, and which
no amount of reading has settled. It prints two values and changes nothing.

The questions, quoted from
[`camera_driver.py`](../../src/assetto_ai_director/application/ports/camera_driver.py):

1. Does `ac.grabCamera` work during **replay playback**, as opposed to live
   driving? Custom Shaders Patch is the only thing that offers absolute camera
   placement at all; Kunos's own Python API has no such call. The one verified
   real-world use of `grabCamera` runs in a live session, and the existing
   automatic replay-directing tool in the community uses a different mechanism
   entirely — which is weak evidence against, and weak evidence is why this
   spike exists rather than an implementation.

2. Can Lua **observe a replay clock**? A camera timeline is keyed to time. If
   the script cannot tell where in the replay it is, the timeline cannot be
   followed, and `cameras.ini` — which switches on track position — remains the
   only output, with everything that costs.

## What hangs on the answer

| Answer | What becomes possible |
|---|---|
| Both yes | A real montage: cuts on time, drone and chase and bonnet cameras alive, `cameras.ini` demoted to a fallback. |
| Camera yes, clock no | Camera control without a timeline. Live direction of a running session; replays still limited to track position. |
| Camera no | `cameras.ini` is the ceiling, permanently. The CSP Lua exporter and the live provider get deleted rather than deferred, and the effort goes into making a position-switched set read as well as it can. |

Until this runs, all three of those are guesses, and the project does not build
on guesses.

## How to run it

1. Copy the `aidirector_spike` folder into
   `<Assetto Corsa>/apps/lua/`.
2. Start Assetto Corsa with Custom Shaders Patch enabled.
3. Load **a replay** — not a live session. That is the whole point; a live
   session answers a question nobody is asking.
4. Open the app from the sidebar. It reports as it goes.
5. Copy what it printed, or send `Documents/Assetto Corsa/logs/`'s CSP log.

The app takes no input and moves the camera by a metre for two seconds, then
gives it straight back. If something else already holds the camera, it says
which script by name rather than failing silently — a grab that is refused looks
exactly like a grab that did nothing.

## What to send back

The four lines it prints under `=== RESULT ===`. That is enough to decide, and
nothing in this folder is worth keeping afterwards.
