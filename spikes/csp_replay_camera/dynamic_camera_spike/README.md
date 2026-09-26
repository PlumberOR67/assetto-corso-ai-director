# Dynamic Camera Spike — Procedural Dynamic Camera Motion System

## Overview
`dynamic_camera_spike` solves the core visual limitation of previous camera spikes: **the camera itself was stationary in 3D space** (only pivoting to track the car).

Inspired by authentic anime and motorsport cinematography (as seen in `FD3S (Takumi) vs R34 (Takumi) PV.mp4`), this spike implements **continuous, procedural 3D camera motion** with dynamic focal length (FOV) transitions, easing curves, and obstacle safety.

---

## 6 Deterministic Motion Profiles

1. **`DOLLY_IN` (Dynamic Push-In)**:
   - Camera pushes towards the oncoming car along track progression ($D_{start} \rightarrow D_{end}$).
   - FOV compresses dynamically ($54^\circ \rightarrow 43^\circ$) using `easeOutQuad`.
   - **Cinematic role:** Heavy braking into corner entries, dramatic apex arrival.

2. **`DOLLY_OUT` (Dynamic Pull-Out / Crane Elevation)**:
   - Camera pulls back and elevates ($11\text{m}$ travel, $+2.2\text{m}$ altitude gain) like a broadcast jib arm.
   - FOV widens ($45^\circ \rightarrow 56^\circ$) to reveal the corner exit and track landscape.
   - **Cinematic role:** Corner exit, power acceleration onto straights.

3. **`SIDE_TRACK` (Parallel / Flank Tracking Vehicle)**:
   - Camera travels along with the car at low elevation ($0.9\text{m}$ above asphalt, $3.6\text{m}$ lateral offset).
   - Dynamic longitudinal drift ($+0.8\text{m} \rightarrow -1.4\text{m}$) creates shifting bodywork perspective.
   - Smooth critically damped interpolation eliminates curb bounce jitter.
   - **Cinematic role:** Mid-corner drift, high-speed flank battle framing.

4. **`CHASE` (Dynamic Follow Cam)**:
   - Dynamic trailing camera ($6.8\text{m}$ behind, $1.25\text{m}$ height).
   - Longitudinal modulation based on acceleration ($\Delta d \propto a_{\text{long}}$): pulls closer during braking, recedes under full throttle.
   - Lateral swing ($\pm 1.2\text{m}$) with `easeInOutCubic`.
   - **Cinematic role:** Acceleration, high-speed cruise, tail-chase battles.

5. **`ORBIT` (Arcing Panoramic Sweeps)**:
   - Camera sweeps a circular arc around the car ($\pm 25^\circ \rightarrow \pm 115^\circ$, radius $7.2\text{m}$).
   - Elevates smoothly ($1.2\text{m}$) through turn apex.
   - **Cinematic role:** Turn-in transitions, dramatic hairpin rotation overview.

6. **`FLY_BY` (Roadside Static/Drift Anchor with Whip-Pan Zoom)**:
   - Ground-anchored roadside placement with subtle backwards drift ($3.0\text{m}$).
   - Look-ahead tracking with reaction delay anticipation.
   - Dynamic zoom pull ($54^\circ \rightarrow 46^\circ$).
   - **Cinematic role:** High-speed fly-past on straights and sweeping bends.

---

## Safety & Collision Rules

- **Ground Clearance:** Per-frame `physics.raycastTrack` downward. If $camY < groundY + 0.5\text{m}$, clamped to $groundY + 0.5\text{m}$.
- **Car Proximity:** Strict distance clamp: $dist(cam, car) \ge 2.2\text{m}$ prevents clipping through vehicle bodywork.
- **Line of Sight:** Real-time raycasting to target point ($carPos + (0, 0.6, 0)$).

---

## Dual UI Modes

### Mode 1: Autonomous Dynamic Director
- Fully automatic telemetry event detection (`BRAKING`, `TURN_IN`, `CORNER`, `APEX`, `EXIT_CORNER`, `ACCELERATION`, `CRUISE`).
- Multi-criteria candidate scoring + automatic motion profile assignment.
- Live HUD displaying active profile, normalized progress bar, real-time FOV, distance to car, and Dutch roll banking angle.

### Mode 2: Manual Profile Tester
- Select any of the 6 profiles via dedicated buttons.
- Sliders for Intensity Multiplier ($0.2\text{x} - 2.5\text{x}$), Duration ($1.0\text{s} - 7.0\text{s}$), Start FOV, and End FOV.
- Flank selector: Left / Right.
- **`TRIGGER PROFILE NOW`** button: immediately activates the selected dynamic motion profile on the live car for instant visual verification!

---

## File Structure

```
spikes/csp_replay_camera/dynamic_camera_spike/
├── dynamic_camera_spike.lua   # Core runtime with 6 motion profiles and telemetry director
├── json.lua                   # JSON serializer
├── manifest.ini               # CSP app manifest
└── README.md                  # This documentation
```
