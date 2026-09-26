-- Event Detection Spike — CSP 0.3.0-preview595
--
-- Real-time driving event and phase detection from ACTUAL replay telemetry.
-- Uses sliding-window kinematic derivatives (speed, dv/dt, yaw rate, curvature)
-- and deterministic rule-based state machine with hysteresis and debounce.
--
-- Supported Driving Phases:
--   - STRAIGHT / CRUISE
--   - APPROACH_CORNER
--   - BRAKING
--   - TURN_IN
--   - CORNER / MID_CORNER
--   - APEX
--   - EXIT_CORNER
--   - ACCELERATION
--
-- Does NOT extrapolate speed into the future.
-- Does NOT modify Timeline Driver, Shot Planner, or aidirector_spike.

local json = require('json')

-- ============================================================================
-- Configuration
-- ============================================================================

local MAX_SAMPLES = 60          -- ~1.8 seconds history at 33 Hz
local SHORT_WINDOW_FRAMES = 6   -- ~0.18s for fast derivatives (instantaneous accel/yaw)
local MED_WINDOW_FRAMES = 14    -- ~0.42s for stable filtered derivatives
local LONG_WINDOW_FRAMES = 24   -- ~0.72s for trend / inflection analysis

local DEFAULT_MIN_CONFIRM_FRAMES = 5 -- ~0.15s confirmation debounce
local FAST_CONFIRM_FRAMES = 2        -- ~0.06s for sudden heavy braking

local DEFAULT_MIN_DURATION_S = 0.30  -- minimum state duration before normal exit
local APEX_MIN_DURATION_S = 0.20     -- apex is inherently transient

-- ============================================================================
-- State
-- ============================================================================

local state = {
  csp = 'unknown',
  probed = false,
  sample_rate = 33.3333,
  track_length_m = 7100.0,

  -- Detection status
  detection_active = false,
  detection_start_frame = 0,
  last_processed_frame = nil,

  -- Telemetry sliding buffer
  samples = {},

  -- Live computed metrics
  live = {
    frame = 0,
    time_s = 0,
    speed_kmh = 0,
    speed_ms = 0,
    accel_ms2 = 0,
    accel_g = 0,
    yaw_rate_dps = 0,
    yaw_accel_dps2 = 0,
    progress = 0,
    lateral_m = 0,
    curv_current = 0,
    curv_ahead = 0,
  },

  -- Current Active Event
  current_event = nil,

  -- Previous Event
  previous_event = nil,

  -- Debounce & Hysteresis State
  pending_state = nil,
  pending_frames = 0,

  -- Recorded Events History
  recorded_events = {},

  -- Runtime API Probes
  api_probes = {},

  -- UI & Messages
  status_message = nil,
  save_message = nil,
  show_diagnostics = false,
  show_recent_log = true,
}

-- ============================================================================
-- Vector Math Utilities
-- ============================================================================

local function v3str(v)
  if v == nil then return 'nil' end
  return string.format('(%.1f, %.1f, %.1f)', v.x, v.y, v.z)
end

local function v3len(v)
  return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
end

local function v3norm(v)
  local len = v3len(v)
  if len < 1e-12 then return vec3(0, 0, 1) end
  return vec3(v.x / len, v.y / len, v.z / len)
end

local function v3dist(a, b)
  local dx = a.x - b.x
  local dy = a.y - b.y
  local dz = a.z - b.z
  return math.sqrt(dx * dx + dy * dy + dz * dz)
end

local function v3dot(a, b)
  return a.x * b.x + a.y * b.y + a.z * b.z
end

-- ============================================================================
-- Runtime API Probing
-- ============================================================================

local function probeAPI(name, getter)
  local ok, val = pcall(getter)
  if not ok then
    return { name = name, status = 'FAILED', err = tostring(val), val_str = 'error' }
  end
  if val == nil then
    return { name = name, status = 'NIL', val_str = 'nil' }
  end
  local val_str = tostring(val)
  if type(val) == 'number' then
    val_str = string.format('%.3f', val)
  elseif type(val) == 'userdata' or type(val) == 'cdata' then
    if val.x and val.y and val.z then
      val_str = string.format('vec3(%.2f, %.2f, %.2f)', val.x, val.y, val.z)
    end
  end
  return { name = name, status = 'AVAILABLE', type = type(val), val_str = val_str }
end

local function runApiDiagnostics()
  local car = ac.getCar(0)
  local sim = ac.getSim()

  state.api_probes = {
    -- Core Replay APIs
    probeAPI('sim.replayCurrentFrame', function() return sim.replayCurrentFrame end),
    probeAPI('sim.replayFrames', function() return sim.replayFrames end),
    probeAPI('sim.replayPlaybackRate', function() return sim.replayPlaybackRate end),
    probeAPI('sim.isReplayActive', function() return sim.isReplayActive end),

    -- Proven Car Kinematics
    probeAPI('car.position', function() return car.position end),
    probeAPI('car.look', function() return car.look end),
    probeAPI('car.side', function() return car.side end),
    probeAPI('car.up', function() return car.up end),
    probeAPI('car.speedKmh', function() return car.speedKmh end),
    probeAPI('car.splinePosition', function() return car.splinePosition end),

    -- Potential Extended Telemetry in Replay
    probeAPI('car.steer', function() return car.steer end),
    probeAPI('car.gas', function() return car.gas end),
    probeAPI('car.brake', function() return car.brake end),
    probeAPI('car.gear', function() return car.gear end),
    probeAPI('car.rpm', function() return car.rpm end),
    probeAPI('car.gForces', function() return car.gForces end),
    probeAPI('car.acceleration', function() return car.acceleration end),
    probeAPI('car.velocity', function() return car.velocity end),
    probeAPI('car.angularVelocity', function() return car.angularVelocity end),
    probeAPI('car.driftAngle', function() return car.driftAngle end),

    -- Track APIs
    probeAPI('ac.worldCoordinateToTrack', function() return ac.worldCoordinateToTrack(car.position) end),
    probeAPI('ac.trackProgressToWorldCoordinate', function() return ac.trackProgressToWorldCoordinate(car.splinePosition or 0) end),
  }
end

-- ============================================================================
-- Track Curvature & Geometry Estimation
-- ============================================================================

local function estimateTrackLength()
  local len = 0
  local prev = ac.trackProgressToWorldCoordinate(0)
  for i = 1, 100 do
    local p = ac.trackProgressToWorldCoordinate(i / 100)
    if prev and p then
      len = len + v3dist(prev, p)
    end
    prev = p
  end
  if len > 500 then
    state.track_length_m = len
  end
  ac.log(string.format('[EventDetector] Track spline length: %.1fm', state.track_length_m))
end

-- Computes horizontal angle difference (curvature) between spline tangent vectors
local function getTrackCurvature(progress, lookaheadOffset)
  local p = progress + (lookaheadOffset or 0)
  if p < 0 then p = p + 1.0 end
  if p > 1 then p = p - 1.0 end

  local p1 = ac.trackProgressToWorldCoordinate(math.max(0.0001, p - 0.0015))
  local p2 = ac.trackProgressToWorldCoordinate(p)
  local p3 = ac.trackProgressToWorldCoordinate(math.min(0.9999, p + 0.0015))
  if not (p1 and p2 and p3) then
    return 0.0
  end

  local v1 = v3norm(vec3(p2.x - p1.x, 0, p2.z - p1.z))
  local v2 = v3norm(vec3(p3.x - p2.x, 0, p3.z - p2.z))
  local dot = math.max(-1.0, math.min(1.0, v3dot(v1, v2)))
  return math.acos(dot) -- angle in radians
end

-- ============================================================================
-- Kinematic Feature Extraction from Sliding Window
-- ============================================================================

local function computeKinematicFeatures()
  local n = #state.samples
  if n < SHORT_WINDOW_FRAMES + 1 then
    return nil
  end

  local s_now = state.samples[n]
  local s_short = state.samples[math.max(1, n - SHORT_WINDOW_FRAMES)]
  local s_med = state.samples[math.max(1, n - MED_WINDOW_FRAMES)]

  local dt_short = math.max(0.01, s_now.t - s_short.t)
  local dt_med = math.max(0.02, s_now.t - s_med.t)

  -- 1. Longitudinal Acceleration (m/s2)
  local a_short = (s_now.speed_ms - s_short.speed_ms) / dt_short
  local a_med = (s_now.speed_ms - s_med.speed_ms) / dt_med
  local a_filtered = 0.4 * a_short + 0.6 * a_med
  local a_g = a_filtered / 9.80665

  -- 2. Horizontal Heading Change & Yaw Rate (deg/s)
  local h_now = v3norm(vec3(s_now.look.x, 0, s_now.look.z))
  local h_short = v3norm(vec3(s_short.look.x, 0, s_short.look.z))

  local dotH = math.max(-1.0, math.min(1.0, v3dot(h_short, h_now)))
  local crossH = h_short.x * h_now.z - h_short.z * h_now.x
  local yaw_rad = math.atan2(crossH, dotH)
  local yaw_deg = yaw_rad * (180.0 / math.pi)
  local yaw_rate_dps = math.abs(yaw_deg) / dt_short
  local signed_yaw_rate = yaw_deg / dt_short

  -- 3. Yaw Rate Acceleration (dω/dt)
  local yaw_accel_dps2 = 0
  if n >= LONG_WINDOW_FRAMES then
    local s_prior = state.samples[n - MED_WINDOW_FRAMES]
    local s_old = state.samples[math.max(1, n - LONG_WINDOW_FRAMES)]
    local dt_prior = math.max(0.02, s_prior.t - s_old.t)
    local h_prior = v3norm(vec3(s_prior.look.x, 0, s_prior.look.z))
    local h_old = v3norm(vec3(s_old.look.x, 0, s_old.look.z))
    local dotP = math.max(-1.0, math.min(1.0, v3dot(h_old, h_prior)))
    local crossP = h_old.x * h_prior.z - h_old.z * h_prior.x
    local prior_yaw_deg = math.abs(math.atan2(crossP, dotP) * (180.0 / math.pi))
    local prior_yaw_rate = prior_yaw_deg / dt_prior
    yaw_accel_dps2 = (yaw_rate_dps - prior_yaw_rate) / dt_short
  end

  -- 4. Track Curvature (Current vs Lookahead ~22m ahead)
  local curv_current = getTrackCurvature(s_now.progress, 0.0)
  local curv_ahead = getTrackCurvature(s_now.progress, 0.0032) -- ~22m on 7100m track

  -- Update live display telemetry
  state.live.frame = s_now.frame
  state.live.time_s = s_now.t
  state.live.speed_kmh = s_now.speed_kmh
  state.live.speed_ms = s_now.speed_ms
  state.live.accel_ms2 = a_filtered
  state.live.accel_g = a_g
  state.live.yaw_rate_dps = yaw_rate_dps
  state.live.yaw_accel_dps2 = yaw_accel_dps2
  state.live.progress = s_now.progress
  state.live.lateral_m = s_now.lateral_m
  state.live.curv_current = curv_current
  state.live.curv_ahead = curv_ahead

  return {
    speed_kmh = s_now.speed_kmh,
    speed_ms = s_now.speed_ms,
    accel = a_filtered,
    accel_g = a_g,
    yaw_rate = yaw_rate_dps,
    signed_yaw_rate = signed_yaw_rate,
    yaw_accel = yaw_accel_dps2,
    curv_current = curv_current,
    curv_ahead = curv_ahead,
    progress = s_now.progress,
    lateral = s_now.lateral_m,
    frame = s_now.frame,
    t = s_now.t,
  }
end

-- ============================================================================
-- Event Rule Evaluation & Confidence Scoring
-- ============================================================================

local function evaluateEventRules(f, prevEvent)
  local prevType = prevEvent and prevEvent.event_type or 'CRUISE'
  local inCorneringSequence = (prevType == 'TURN_IN' or prevType == 'CORNER' or prevType == 'APEX')

  local proposals = {}

  -- 1. BRAKING: Strong negative longitudinal acceleration
  if f.accel < -2.0 then
    local conf = math.min(0.98, 0.60 + (math.abs(f.accel) - 2.0) * 0.09)
    if f.accel < -4.2 then conf = math.max(0.88, conf) end
    proposals['BRAKING'] = conf
  end

  -- 2. ACCELERATION: Positive longitudinal acceleration on exit/straight
  if f.accel > 1.1 and f.yaw_rate < 15.0 then
    local conf = math.min(0.96, 0.62 + (f.accel - 1.1) * 0.12)
    if prevType == 'EXIT_CORNER' then conf = conf + 0.05 end
    proposals['ACCELERATION'] = math.min(0.98, conf)
  end

  -- 3. TURN_IN: Rapid rise in yaw rate entering corner
  if f.yaw_rate > 11.0 and f.yaw_accel > 6.0 and f.accel < 1.2 then
    local conf = math.min(0.93, 0.65 + (f.yaw_rate - 11.0) * 0.015)
    if prevType == 'BRAKING' or prevType == 'APPROACH_CORNER' or prevType == 'CRUISE' then
      conf = conf + 0.06
    end
    proposals['TURN_IN'] = math.min(0.95, conf)
  end

  -- 4. APEX: Inflection / peak turn / minimum speed moment inside corner
  if inCorneringSequence and prevEvent.duration >= 0.30 then
    -- Near peak yaw rate or speed reaching local inflection
    local isYawPeak = (f.yaw_rate > 14.0 and f.yaw_accel <= 3.0)
    local isSpeedMin = (f.accel >= -1.0 and f.accel <= 0.8 and f.yaw_rate > 13.0)
    if (isYawPeak or isSpeedMin) and prevType ~= 'APEX' and prevType ~= 'EXIT_CORNER' then
      local conf = 0.82
      if isYawPeak and isSpeedMin then conf = 0.90 end
      proposals['APEX'] = conf
    end
  end

  -- 5. CORNER / MID_CORNER: Sustained turning rate
  if (f.yaw_rate > 14.0 or (f.curv_current > 0.038 and f.yaw_rate > 9.0)) and f.accel < 1.6 then
    local conf = math.min(0.94, 0.68 + (f.yaw_rate - 14.0) * 0.012)
    if inCorneringSequence then conf = conf + 0.04 end
    proposals['CORNER'] = math.min(0.96, conf)
  end

  -- 6. EXIT_CORNER: Yaw rate declining out of corner
  if (prevType == 'APEX' or prevType == 'CORNER') and f.yaw_rate < 13.0 and f.yaw_accel < 0.0 and f.accel > -1.0 then
    local conf = math.min(0.88, 0.72 + (13.0 - f.yaw_rate) * 0.015)
    proposals['EXIT_CORNER'] = conf
  end

  -- 7. APPROACH_CORNER: Straight road with upcoming sharp curvature ahead
  if f.curv_ahead > 0.045 and f.curv_current < 0.025 and f.yaw_rate < 7.5 and f.speed_kmh > 45.0 and f.accel > -2.0 then
    local conf = math.min(0.86, 0.65 + (f.curv_ahead - 0.045) * 3.5)
    proposals['APPROACH_CORNER'] = conf
  end

  -- 8. CRUISE / STRAIGHT: Low turning, stable speed
  if f.yaw_rate < 6.5 and math.abs(f.accel) < 1.1 and f.curv_ahead < 0.035 then
    local conf = math.min(0.95, 0.82 + (6.5 - f.yaw_rate) * 0.018)
    proposals['CRUISE'] = conf
  end

  -- Select the proposal with highest confidence
  local bestType = 'CRUISE'
  local bestConf = 0.50

  for etype, conf in pairs(proposals) do
    if conf > bestConf then
      bestType = etype
      bestConf = conf
    end
  end

  -- Fallback if no specific rule matched
  if bestConf <= 0.50 then
    if f.yaw_rate > 12.0 then
      bestType = 'CORNER'
      bestConf = 0.60
    elseif f.accel > 0.8 then
      bestType = 'ACCELERATION'
      bestConf = 0.62
    elseif f.accel < -1.2 then
      bestType = 'BRAKING'
      bestConf = 0.62
    else
      bestType = 'CRUISE'
      bestConf = 0.70
    end
  end

  return bestType, math.floor(bestConf * 100) / 100
end

-- ============================================================================
-- Event State Machine & Hysteresis Transitions
-- ============================================================================

local function startNewEvent(etype, conf, frame, speedKmh, progress, features)
  state.current_event = {
    index = #state.recorded_events + 1,
    event_type = etype,
    confidence = conf,
    start_frame = frame,
    end_frame = frame,
    start_time = math.floor((frame / state.sample_rate) * 100) / 100,
    end_time = math.floor((frame / state.sample_rate) * 100) / 100,
    duration = 0.0,
    speed_start = math.floor(speedKmh * 10) / 10,
    speed_end = math.floor(speedKmh * 10) / 10,
    progress_start = math.floor(progress * 10000) / 10000,
    progress_end = math.floor(progress * 10000) / 10000,
    metrics = {
      yaw_rates = { features and features.yaw_rate or 0 },
      accels = { features and features.accel or 0 },
    },
  }
end

local function finalizeCurrentEvent(endFrame, speedKmh, progress)
  if state.current_event == nil then return end

  local ev = state.current_event
  local actualEnd = math.max(ev.start_frame, endFrame)
  ev.end_frame = actualEnd
  ev.end_time = math.floor((actualEnd / state.sample_rate) * 100) / 100
  local fCount = actualEnd - ev.start_frame + 1
  ev.duration = math.floor((fCount / state.sample_rate) * 100) / 100
  ev.speed_end = math.floor(speedKmh * 10) / 10
  ev.progress_end = math.floor(progress * 10000) / 10000

  -- Aggregate metrics summary
  local sumYaw, maxYaw = 0, 0
  for _, yr in ipairs(ev.metrics.yaw_rates) do
    sumYaw = sumYaw + yr
    if yr > maxYaw then maxYaw = yr end
  end
  local avgYaw = #ev.metrics.yaw_rates > 0 and (sumYaw / #ev.metrics.yaw_rates) or 0

  local sumAcc, minAcc, maxAcc = 0, 999, -999
  for _, acVal in ipairs(ev.metrics.accels) do
    sumAcc = sumAcc + acVal
    if acVal < minAcc then minAcc = acVal end
    if acVal > maxAcc then maxAcc = acVal end
  end
  if minAcc == 999 then minAcc = 0 end
  if maxAcc == -999 then maxAcc = 0 end
  local avgAcc = #ev.metrics.accels > 0 and (sumAcc / #ev.metrics.accels) or 0

  local headingMetrics = {
    avg_yaw_rate_dps = math.floor(avgYaw * 10) / 10,
    max_yaw_rate_dps = math.floor(maxYaw * 10) / 10,
    avg_accel_ms2 = math.floor(avgAcc * 10) / 10,
    min_accel_ms2 = math.floor(minAcc * 10) / 10,
    max_accel_ms2 = math.floor(maxAcc * 10) / 10,
    speed_delta_kmh = math.floor((ev.speed_end - ev.speed_start) * 10) / 10,
  }

  local record = {
    index = ev.index,
    event_type = ev.event_type,
    confidence = ev.confidence,
    start_frame = ev.start_frame,
    end_frame = ev.end_frame,
    start_time = ev.start_time,
    end_time = ev.end_time,
    duration = ev.duration,
    speed_start = ev.speed_start,
    speed_end = ev.speed_end,
    progress_start = ev.progress_start,
    progress_end = ev.progress_end,
    heading_metrics = headingMetrics,
  }

  state.recorded_events[#state.recorded_events + 1] = record
  state.previous_event = record

  ac.log(string.format('[EventDetector] #%02d %-15s (%.2fs | conf: %.2f | spd: %.1f->%.1f km/h | max_yaw: %.1f dps)',
    record.index, record.event_type, record.duration, record.confidence,
    record.speed_start, record.speed_end, headingMetrics.max_yaw_rate_dps))
end

-- ============================================================================
-- Core Update Loop
-- ============================================================================

local function updateEventDetector()
  if not state.detection_active then return end

  local sim = ac.getSim()
  if sim == nil then return end

  local currentFrame = sim.replayCurrentFrame or 0
  local isPaused = (sim.replayPlaybackRate == 0)

  local car = ac.getCar(0)
  if car == nil or car.position == nil then return end

  local carPos = vec3(car.position.x, car.position.y, car.position.z)
  local carLook = car.look and vec3(car.look.x, car.look.y, car.look.z) or vec3(0, 0, 1)
  local carSpeed = car.speedKmh or 50.0
  local carProgress = car.splinePosition or 0

  -- Replay scrub / jump detection
  local scrubDetected = false
  if state.last_processed_frame ~= nil then
    if math.abs(currentFrame - state.last_processed_frame) > 40 then
      scrubDetected = true
      ac.log(string.format('[EventDetector] Scrub detected (%d -> %d). Resetting window.',
        state.last_processed_frame, currentFrame))
      state.samples = {}
      if state.current_event then
        finalizeCurrentEvent(state.last_processed_frame, carSpeed, carProgress)
        state.current_event = nil
      end
    end
  end

  -- Skip processing if paused or same frame
  if not scrubDetected and (isPaused or currentFrame == state.last_processed_frame) then
    return
  end

  state.last_processed_frame = currentFrame

  -- Record sample into sliding buffer
  local sampleT = currentFrame / state.sample_rate
  local lateral_m = 0
  local wtOk, trk = pcall(function() return ac.worldCoordinateToTrack(carPos) end)
  if wtOk and trk then lateral_m = trk.x end

  table.insert(state.samples, {
    frame = currentFrame,
    t = sampleT,
    pos = carPos,
    look = carLook,
    speed_kmh = carSpeed,
    speed_ms = carSpeed / 3.6,
    progress = carProgress,
    lateral_m = lateral_m,
  })

  while #state.samples > MAX_SAMPLES do
    table.remove(state.samples, 1)
  end

  -- Compute derivatives
  local features = computeKinematicFeatures()
  if features == nil then return end

  -- Evaluate rule proposals
  local proposedState, proposedConf = evaluateEventRules(features, state.current_event)

  -- Initial event bootstrap
  if state.current_event == nil then
    startNewEvent(proposedState, proposedConf, currentFrame, carSpeed, carProgress, features)
    return
  end

  local curEv = state.current_event
  local curDuration = (currentFrame - curEv.start_frame + 1) / state.sample_rate

  -- Hysteresis / Debounce state machine
  if proposedState == curEv.event_type then
    -- Candidate matches current state: remain steady
    state.pending_state = nil
    state.pending_frames = 0
    curEv.duration = math.floor(curDuration * 100) / 100
    curEv.end_frame = currentFrame
    curEv.confidence = math.floor((0.92 * curEv.confidence + 0.08 * proposedConf) * 100) / 100
    table.insert(curEv.metrics.yaw_rates, features.yaw_rate)
    table.insert(curEv.metrics.accels, features.accel)
  else
    -- Candidate wants to switch
    if proposedState == state.pending_state then
      state.pending_frames = state.pending_frames + 1
    else
      state.pending_state = proposedState
      state.pending_frames = 1
    end

    -- Required confirmation threshold
    local reqFrames = DEFAULT_MIN_CONFIRM_FRAMES
    local minDur = (curEv.event_type == 'APEX') and APEX_MIN_DURATION_S or DEFAULT_MIN_DURATION_S

    -- Emergency fast override for heavy braking
    if proposedState == 'BRAKING' and features.accel < -4.0 then
      reqFrames = FAST_CONFIRM_FRAMES
      minDur = 0.15
    end

    -- Confirm transition once stability thresholds are satisfied
    if state.pending_frames >= reqFrames and curDuration >= minDur then
      finalizeCurrentEvent(currentFrame - 1, carSpeed, carProgress)
      startNewEvent(proposedState, proposedConf, currentFrame, carSpeed, carProgress, features)
      state.pending_state = nil
      state.pending_frames = 0
    else
      -- Still debouncing, maintain current state
      curEv.duration = math.floor(curDuration * 100) / 100
      curEv.end_frame = currentFrame
      table.insert(curEv.metrics.yaw_rates, features.yaw_rate)
      table.insert(curEv.metrics.accels, features.accel)
    end
  end
end

-- ============================================================================
-- Start / Stop Detection Session
-- ============================================================================

local function startDetection()
  if state.detection_active then return end

  state.detection_active = true
  local sim = ac.getSim()
  state.detection_start_frame = sim and sim.replayCurrentFrame or 0
  state.last_processed_frame = nil
  state.samples = {}
  state.recorded_events = {}
  state.current_event = nil
  state.previous_event = nil
  state.pending_state = nil
  state.pending_frames = 0
  state.save_message = nil
  state.status_message = 'Event Detection: RUNNING'
  ac.log('[EventDetector] Started detection session at frame ' .. state.detection_start_frame)
end

local function stopDetection()
  if not state.detection_active then return end

  local sim = ac.getSim()
  local currentFrame = sim and sim.replayCurrentFrame or (state.last_processed_frame or 0)
  local car = ac.getCar(0)
  local spd = car and car.speedKmh or 0
  local prog = car and car.splinePosition or 0

  if state.current_event then
    finalizeCurrentEvent(currentFrame, spd, prog)
    state.current_event = nil
  end

  state.detection_active = false
  state.status_message = string.format('Detection stopped. %d events recorded.', #state.recorded_events)
  ac.log('[EventDetector] Stopped. Total events: ' .. #state.recorded_events)
end

-- ============================================================================
-- Save Event Detection Report JSON
-- ============================================================================

local function saveEventReport()
  if #state.recorded_events == 0 then
    state.save_message = 'Error: no events recorded yet. Run replay and test first.'
    return false
  end

  local docsOk, docs = pcall(function() return ac.getFolder(ac.FolderID.ACDocuments) end)
  if not docsOk or docs == nil then
    state.save_message = 'Error: cannot locate Documents folder'
    return false
  end

  local path = docs .. '/event_detection_report.json'

  -- Calculate summary stats
  local startF = state.recorded_events[1].start_frame
  local endF = state.recorded_events[#state.recorded_events].end_frame
  local totalDur = math.floor(((endF - startF + 1) / state.sample_rate) * 100) / 100

  local eventsByType = {}
  local sumConf = 0
  local shortCount = 0

  for _, ev in ipairs(state.recorded_events) do
    eventsByType[ev.event_type] = (eventsByType[ev.event_type] or 0) + 1
    sumConf = sumConf + (ev.confidence or 0)
    if ev.duration < 0.30 then
      shortCount = shortCount + 1
    end
  end

  local avgConf = #state.recorded_events > 0 and (math.floor((sumConf / #state.recorded_events) * 100) / 100) or 0

  local report = {
    schema_version = 1,
    app = 'event_detection_spike',
    csp_version = state.csp,
    timestamp = os.time(),
    sample_rate_hz = state.sample_rate,
    track_length_m = state.track_length_m,
    summary = {
      replay_range_analysed = {
        start_frame = startF,
        end_frame = endF,
        duration_s = totalDur,
      },
      total_events = #state.recorded_events,
      events_by_type = eventsByType,
      average_confidence = avgConf,
      short_events_count = shortCount,
      short_events_ratio_pct = math.floor((shortCount / #state.recorded_events) * 1000) / 10,
      api_diagnostics = state.api_probes,
    },
    events = state.recorded_events,
  }

  local encOk, encoded = pcall(function() return json.encode(report) end)
  if not encOk or encoded == nil then
    state.save_message = 'Error encoding JSON report: ' .. tostring(encoded)
    return false
  end

  local file = io.open(path, 'w')
  if file == nil then
    state.save_message = 'Error opening file: ' .. path
    return false
  end

  file:write(encoded)
  file:close()

  state.save_message = string.format('Report saved: %s (%d events, %.1fs)', path, #state.recorded_events, totalDur)
  ac.log('[EventDetector] Saved report: ' .. path)
  return true
end

-- ============================================================================
-- UI Window Rendering
-- ============================================================================

local function drawWindow()
  ui.text('=== EVENT DETECTION SPIKE ===')
  ui.text('CSP: ' .. state.csp)

  -- Telemetry Header
  local sim = ac.getSim()
  local currentFrame = sim and sim.replayCurrentFrame or 0
  local replayTime = currentFrame / state.sample_rate

  ui.text(string.format('Replay: frame %d | t = %.2fs | rate = %.1fx',
    currentFrame, replayTime, sim and sim.replayPlaybackRate or 1.0))
  ui.text(string.format('Car: Spd %.1f km/h (%.1f m/s) | Progress: %.4f',
    state.live.speed_kmh, state.live.speed_ms, state.live.progress))

  ui.separator()

  -- Action Controls
  if not state.detection_active then
    if ui.button('START EVENT DETECTION', vec2(220, 32)) then
      startDetection()
    end
    ui.sameLine()
    if #state.recorded_events > 0 then
      if ui.button('Save Event Report', vec2(150, 32)) then
        saveEventReport()
      end
      ui.sameLine()
      if ui.button('Clear', vec2(80, 32)) then
        state.recorded_events = {}
        state.previous_event = nil
        state.current_event = nil
        state.save_message = nil
        state.status_message = 'Cleared event buffer.'
      end
    end
  else
    if ui.button('STOP EVENT DETECTION', vec2(220, 32)) then
      stopDetection()
    end
    ui.sameLine()
    if #state.recorded_events > 0 then
      if ui.button('Save Event Report', vec2(150, 32)) then
        saveEventReport()
      end
    end
  end

  if state.save_message then
    ui.textWrapped('[SAVED] ' .. state.save_message)
  end
  if state.status_message then
    ui.text(state.status_message)
  end

  ui.separator()

  -- Live Kinematic Derivatives
  ui.text('--- Kinematic Derivatives (Sliding Window) ---')
  ui.text(string.format('Accel:  %+.2f m/s²  (%+.2f G)', state.live.accel_ms2, state.live.accel_g))
  ui.text(string.format('Yaw:    %.1f °/s (dω/dt: %+.1f °/s²)', state.live.yaw_rate_dps, state.live.yaw_accel_dps2))
  ui.text(string.format('Curv:   current %.3f rad | ahead: %.3f rad', state.live.curv_current, state.live.curv_ahead))

  ui.separator()

  -- Active Event Status Card
  if state.detection_active and state.current_event then
    local ev = state.current_event
    ui.text('=== CURRENT EVENT ===')
    ui.text(string.format('EVENT:      %s', ev.event_type))
    ui.text(string.format('CONFIDENCE: %.2f', ev.confidence))
    ui.text(string.format('START TIME: %.2fs (frame %d)', ev.start_time, ev.start_frame))
    ui.text(string.format('DURATION:   %.2fs', ev.duration))
    ui.text(string.format('SPEED:      %.1f km/h (entry)', ev.speed_start))
  elseif not state.detection_active and #state.recorded_events > 0 then
    ui.text('Detection paused. Click START EVENT DETECTION to record more.')
  else
    ui.text('Event detector idle. Press START EVENT DETECTION to begin.')
  end

  -- Previous Event Card
  if state.previous_event then
    ui.separator()
    local pv = state.previous_event
    ui.text('--- PREVIOUS EVENT ---')
    ui.text(string.format('%s  [%.2fs - %.2fs | dur: %.2fs | conf: %.2f]',
      pv.event_type, pv.start_time, pv.end_time, pv.duration, pv.confidence))
    ui.text(string.format('Spd: %.1f -> %.1f km/h | Max Yaw: %.1f °/s',
      pv.speed_start, pv.speed_end, pv.heading_metrics.max_yaw_rate_dps))
  end

  -- Summary Statistics
  if #state.recorded_events > 0 then
    ui.separator()
    ui.text(string.format('=== RECORDED EVENTS SUMMARY (%d total) ===', #state.recorded_events))

    local counts = {}
    local shortCount = 0
    for _, ev in ipairs(state.recorded_events) do
      counts[ev.event_type] = (counts[ev.event_type] or 0) + 1
      if ev.duration < 0.30 then shortCount = shortCount + 1 end
    end

    local breakdown = {}
    for k, v in pairs(counts) do
      table.insert(breakdown, string.format('%s: %d', k, v))
    end
    ui.textWrapped(table.concat(breakdown, ' | '))
    ui.text(string.format('Short events (<0.30s): %d (%.1f%% of total)',
      shortCount, (shortCount / #state.recorded_events) * 100))

    -- Recent Event History List
    ui.separator()
    ui.text('=== RECENT EVENTS TIMELINE ===')
    local startIdx = math.max(1, #state.recorded_events - 11)
    for i = #state.recorded_events, startIdx, -1 do
      local r = state.recorded_events[i]
      local spdStr = string.format('%.0f->%.0f km/h', r.speed_start, r.speed_end)
      ui.text(string.format('%02d. %05.1fs  %-14s (%.2fs, conf:%.2f, %s, yaw:%.0f°/s)',
        r.index, r.start_time, r.event_type, r.duration, r.confidence, spdStr, r.heading_metrics.max_yaw_rate_dps))
    end
  end

  -- Telemetry API Diagnostics Panel
  ui.separator()
  local diagLabel = state.show_diagnostics and '[-] Hide Runtime API Diagnostics' or '[+] Show Runtime API Diagnostics'
  if ui.button(diagLabel, vec2(280, 24)) then
    state.show_diagnostics = not state.show_diagnostics
  end

  if state.show_diagnostics then
    ui.text('--- Probed CSP Runtime APIs ---')
    for _, p in ipairs(state.api_probes) do
      local tag = (p.status == 'AVAILABLE') and '[OK]' or '[--]'
      ui.text(string.format('%-4s %-28s = %s', tag, p.name, p.val_str))
    end
  end
end

-- ============================================================================
-- Script Lifecycle Entry Point
-- ============================================================================

function script.windowMain()
  if not state.probed then
    state.probed = true
    local ok, v = pcall(function() return ac.getPatchVersion() end)
    state.csp = ok and tostring(v) or 'unknown'
    estimateTrackLength()
    runApiDiagnostics()
  end

  updateEventDetector()
  drawWindow()
end
