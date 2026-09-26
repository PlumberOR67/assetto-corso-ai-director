-- Director Integration Spike — CSP 0.3.0-preview595
--
-- Event-Aware Cinematic Replay Director
-- Integrates real-time replay telemetry, macro corner sequence coalescing with Apex Latch,
-- event-priority camera style policies, multi-criteria scoring, and adaptive pacing cut triggers.
--
-- Does NOT modify Timeline Driver, Shot Planner, or aidirector_spike.
-- Does NOT extrapolate speed into the future.
-- Does NOT generate timeline.json yet.

local json = require('json')

-- ============================================================================
-- Configuration & Calibrated Parameters
-- ============================================================================

local RAISE_HEIGHT = 80.0
local RAY_LENGTH = 200.0
local MIN_CLEARANCE = 0.5
local LOS_TOLERANCE = 2.0

-- 10 Calibrated Camera Types (proven 25/26 visual clean rate)
local CAMERA_TYPES = {
  { name = 'roadside_left_close',   lateral = -1.25, height = 1.4, category = 'roadside', side = 'left',   desc = 'Left edge, eye level' },
  { name = 'roadside_right_close',  lateral =  1.25, height = 1.4, category = 'roadside', side = 'right',  desc = 'Right edge, eye level' },
  { name = 'roadside_left_wide',    lateral = -1.65, height = 1.6, category = 'wide',     side = 'left',   desc = 'Wide left berm' },
  { name = 'roadside_right_wide',   lateral =  1.65, height = 1.6, category = 'wide',     side = 'right',  desc = 'Wide right berm' },
  { name = 'low_left',              lateral = -1.15, height = 0.6, category = 'low',      side = 'left',   desc = 'Low left curb view' },
  { name = 'low_right',             lateral =  1.15, height = 0.6, category = 'low',      side = 'right',  desc = 'Low right curb view' },
  { name = 'elevated_left_low',     lateral = -1.35, height = 2.8, category = 'elevated', side = 'left',   desc = 'Elevated left, over bushes (2.8m)' },
  { name = 'elevated_left_mid',     lateral = -1.45, height = 4.5, category = 'elevated', side = 'left',   desc = 'Elevated left boom pole (4.5m)' },
  { name = 'elevated_right',        lateral =  1.30, height = 3.5, category = 'elevated', side = 'right',  desc = 'Elevated right overview (3.5m)' },
  { name = 'elevated_high',         lateral = -0.80, height = 7.5, category = 'high',     side = 'center', desc = 'High crane / overhead (7.5m)' },
}

-- Tight Progress Offsets: -30m to +40m focal zone
local PROGRESS_OFFSETS = { -0.0040, -0.0020, 0.0, 0.0020, 0.0040, 0.0055 }

-- Pacing & Cut Duration Thresholds
local MIN_SHOT_DURATION_DEFAULT = 1.8   -- seconds
local MIN_SHOT_DURATION_APEX = 1.2      -- seconds
local PREFERRED_SHOT_DURATION = 2.8     -- seconds
local MAX_SHOT_DURATION_DEFAULT = 4.2   -- seconds
local MAX_SHOT_DURATION_CRUISE = 5.2    -- seconds
local CUT_COOLDOWN_S = 1.0              -- seconds
local SCORE_IMPROVEMENT_THRESHOLD = 18.0 -- points needed for early candidate cut

local DEFAULT_MIN_CONFIRM_FRAMES = 5    -- ~0.15s confirmation debounce
local FAST_CONFIRM_FRAMES = 2           -- ~0.06s for sudden heavy braking
local DEFAULT_MIN_DURATION_S = 0.35     -- minimum event duration before normal exit
local APEX_MIN_DURATION_S = 0.20        -- apex is inherently transient

-- ============================================================================
-- State
-- ============================================================================

local state = {
  csp = 'unknown',
  probed = false,
  sample_rate = 33.3333,
  track_length_m = 7100.0,

  -- Director runtime status
  director_active = false,
  director_cam = nil,
  session_start_frame = 0,
  last_processed_frame = nil,

  -- Telemetry sliding buffer
  samples = {},
  max_samples = 60,

  -- Live telemetry
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

  -- Driving Phase & Macro Sequence State
  driving_event = {
    event_type = 'CRUISE',
    confidence = 0.90,
    importance = 0.50,
    start_time = 0,
    start_frame = 0,
    duration = 0,
  },
  macro_state = 'STRAIGHT_RUN',  -- 'STRAIGHT_RUN' or 'CORNER_SEQUENCE'
  apex_latched = false,           -- Latch: at most ONE apex shot per corner sequence!

  -- Event Hysteresis & Visibility Stabilization
  pending_state = nil,
  pending_frames = 0,
  los_occluded_frames = 0,

  -- Current Active Shot
  current_shot = nil,
  shot_start_frame = 0,
  shot_start_time = 0,
  last_cut_reason = 'INIT',

  -- Candidate evaluation counters
  eval_count = 0,
  eval_accepted = 0,
  eval_rejected = 0,

  -- Session Recording
  session_shots = {},

  -- UI & Diagnostics
  status_message = nil,
  save_message = nil,
  show_diagnostics = false,
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

local function v3cross(a, b)
  return vec3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  )
end

-- ============================================================================
-- Raycast Physics Validation
-- ============================================================================

local function findGround(worldPos)
  local origin = vec3(worldPos.x, worldPos.y + RAISE_HEIGHT, worldPos.z)
  local ok, ray = pcall(function()
    return render.createRay(origin, vec3(0, -1, 0), RAY_LENGTH)
  end)
  if not ok or ray == nil then return nil, nil end

  local outPos = vec3(9e9, 9e9, 9e9)
  local outNormal = vec3(0, 0, 0)
  local physOk, dist = pcall(function()
    return ray:physics(outPos, outNormal)
  end)
  pcall(function() ray:dispose() end)

  if not physOk then return nil, nil end
  if type(dist) ~= 'number' or dist < 0 then return nil, nil end

  local groundY = origin.y - dist
  if outPos.y ~= 9e9 then
    groundY = outPos.y
  end

  return groundY, dist
end

local function checkLineOfSight(camPos, targetPos)
  local dir = vec3(targetPos.x - camPos.x, targetPos.y - camPos.y, targetPos.z - camPos.z)
  local totalDist = v3len(dir)
  if totalDist < 0.1 then return true, -1, totalDist end

  local dirNorm = v3norm(dir)

  local ok, ray = pcall(function()
    return render.createRay(camPos, dirNorm, totalDist + 1.0)
  end)
  if not ok or ray == nil then return false, -1, totalDist end

  local outPos = vec3(9e9, 9e9, 9e9)
  local outNormal = vec3(0, 0, 0)
  local physOk, dist = pcall(function()
    return ray:physics(outPos, outNormal)
  end)
  pcall(function() ray:dispose() end)

  if not physOk then return false, -1, totalDist end
  if type(dist) ~= 'number' or dist < 0 then
    return true, -1, totalDist
  end

  if dist >= totalDist - LOS_TOLERANCE then
    return true, dist, totalDist
  end

  return false, dist, totalDist
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
  ac.log(string.format('[DirectorIntegration] Track length: %.1fm', state.track_length_m))
end

local function getTrackCurvature(progress, lookaheadOffset)
  local p = progress + (lookaheadOffset or 0)
  if p < 0 then p = p + 1.0 end
  if p > 1 then p = p - 1.0 end

  local p1 = ac.trackProgressToWorldCoordinate(math.max(0.0001, p - 0.0015))
  local p2 = ac.trackProgressToWorldCoordinate(p)
  local p3 = ac.trackProgressToWorldCoordinate(math.min(0.9999, p + 0.0015))
  if not (p1 and p2 and p3) then return 0.0 end

  local v1 = v3norm(vec3(p2.x - p1.x, 0, p2.z - p1.z))
  local v2 = v3norm(vec3(p3.x - p2.x, 0, p3.z - p2.z))
  local dot = math.max(-1.0, math.min(1.0, v3dot(v1, v2)))
  return math.acos(dot)
end

-- ============================================================================
-- Kinematic Derivative Computation from Sliding Buffer
-- ============================================================================

local function computeKinematics()
  local n = #state.samples
  if n < 8 then return nil end

  local s_now = state.samples[n]
  local s_short = state.samples[math.max(1, n - 6)]
  local s_med = state.samples[math.max(1, n - 14)]

  local dt_short = math.max(0.01, s_now.t - s_short.t)
  local dt_med = math.max(0.02, s_now.t - s_med.t)

  -- Longitudinal acceleration
  local a_short = (s_now.speed_ms - s_short.speed_ms) / dt_short
  local a_med = (s_now.speed_ms - s_med.speed_ms) / dt_med
  local a_filtered = 0.4 * a_short + 0.6 * a_med
  local a_g = a_filtered / 9.80665

  -- Heading change & yaw rate
  local h_now = v3norm(vec3(s_now.look.x, 0, s_now.look.z))
  local h_short = v3norm(vec3(s_short.look.x, 0, s_short.look.z))
  local dotH = math.max(-1.0, math.min(1.0, v3dot(h_short, h_now)))
  local crossH = h_short.x * h_now.z - h_short.z * h_now.x
  local yaw_deg = math.atan2(crossH, dotH) * (180.0 / math.pi)
  local yaw_rate = math.abs(yaw_deg) / dt_short

  -- Yaw acceleration (dω/dt)
  local yaw_accel = 0
  if n >= 22 then
    local s_prior = state.samples[n - 14]
    local s_old = state.samples[math.max(1, n - 22)]
    local dt_p = math.max(0.02, s_prior.t - s_old.t)
    local h_p = v3norm(vec3(s_prior.look.x, 0, s_prior.look.z))
    local h_o = v3norm(vec3(s_old.look.x, 0, s_old.look.z))
    local dotP = math.max(-1.0, math.min(1.0, v3dot(h_o, h_p)))
    local crossP = h_o.x * h_p.z - h_o.z * h_p.x
    local prior_yaw_rate = math.abs(math.atan2(crossP, dotP) * (180.0 / math.pi)) / dt_p
    yaw_accel = (yaw_rate - prior_yaw_rate) / dt_short
  end

  local curv_current = getTrackCurvature(s_now.progress, 0.0)
  local curv_ahead = getTrackCurvature(s_now.progress, 0.0032)

  state.live.frame = s_now.frame
  state.live.time_s = s_now.t
  state.live.speed_kmh = s_now.speed_kmh
  state.live.speed_ms = s_now.speed_ms
  state.live.accel_ms2 = a_filtered
  state.live.accel_g = a_g
  state.live.yaw_rate_dps = yaw_rate
  state.live.yaw_accel_dps2 = yaw_accel
  state.live.progress = s_now.progress
  state.live.lateral_m = s_now.lateral_m
  state.live.curv_current = curv_current
  state.live.curv_ahead = curv_ahead

  return {
    frame = s_now.frame,
    t = s_now.t,
    speed_kmh = s_now.speed_kmh,
    speed_ms = s_now.speed_ms,
    accel = a_filtered,
    accel_g = a_g,
    yaw_rate = yaw_rate,
    yaw_accel = yaw_accel,
    curv_current = curv_current,
    curv_ahead = curv_ahead,
    progress = s_now.progress,
    lateral = s_now.lateral_m,
  }
end

-- ============================================================================
-- Event Importance Calculation (Measurable 0.10 to 1.00)
-- ============================================================================

local function calculateEventImportance(etype, f)
  local imp = 0.50

  if etype == 'BRAKING' then
    local decel = math.abs(math.min(0, f.accel))
    imp = 0.45 + math.min(0.48, (decel / 8.0) * 0.48)
    if f.speed_kmh > 100.0 then imp = imp + 0.05 end

  elseif etype == 'APEX' then
    imp = 0.76 + math.min(0.20, (f.yaw_rate / 60.0) * 0.20)

  elseif etype == 'TURN_IN' then
    local yawAcc = math.max(0, f.yaw_accel)
    imp = 0.55 + math.min(0.35, (f.yaw_rate / 50.0) * 0.20 + (yawAcc / 30.0) * 0.15)

  elseif etype == 'CORNER' then
    imp = 0.52 + math.min(0.38, (f.yaw_rate / 60.0) * 0.38)
    if f.speed_kmh > 70.0 then imp = imp + 0.06 end

  elseif etype == 'EXIT_CORNER' then
    imp = 0.55 + math.min(0.25, (f.speed_kmh / 120.0) * 0.25)

  elseif etype == 'ACCELERATION' then
    local accVal = math.max(0, f.accel)
    imp = 0.46 + math.min(0.44, (accVal / 5.5) * 0.35 + (f.speed_kmh / 150.0) * 0.09)

  elseif etype == 'APPROACH_CORNER' then
    imp = 0.48 + math.min(0.30, (f.curv_ahead / 0.08) * 0.30)

  elseif etype == 'CRUISE' then
    imp = 0.35 + math.min(0.25, (f.speed_kmh / 160.0) * 0.25)
  end

  return math.max(0.10, math.min(1.0, math.floor(imp * 100 + 0.5) / 100))
end

-- ============================================================================
-- Event Rule Evaluation & Confidence Scoring (Proven from event_detection_spike)
-- ============================================================================

local function evaluateEventRules(f, curEv)
  local prevType = curEv and curEv.event_type or 'CRUISE'
  local inCorneringSequence = (state.macro_state == 'CORNER_SEQUENCE') or
                             (prevType == 'TURN_IN' or prevType == 'CORNER' or prevType == 'APEX')

  local proposals = {}

  -- 1. BRAKING: Strong negative longitudinal acceleration
  if f.accel < -2.0 then
    local conf = math.min(0.98, 0.60 + (math.abs(f.accel) - 2.0) * 0.09)
    if f.accel < -4.2 then conf = math.max(0.88, conf) end
    proposals['BRAKING'] = conf
  end

  -- 2. ACCELERATION: Positive longitudinal acceleration on exit/straight
  if f.accel > 1.0 and f.yaw_rate < 15.0 then
    local conf = math.min(0.96, 0.62 + (f.accel - 1.0) * 0.12)
    if prevType == 'EXIT_CORNER' then conf = conf + 0.05 end
    proposals['ACCELERATION'] = math.min(0.98, conf)
  end

  -- 3. TURN_IN: Rapid rise in yaw rate entering corner
  if f.yaw_rate > 11.0 and f.yaw_accel > 5.0 and f.accel < 1.2 then
    local conf = math.min(0.93, 0.65 + (f.yaw_rate - 11.0) * 0.015)
    if prevType == 'BRAKING' or prevType == 'APPROACH_CORNER' or prevType == 'CRUISE' then
      conf = conf + 0.06
    end
    proposals['TURN_IN'] = math.min(0.95, conf)
  end

  -- 4. APEX: Inflection / peak turn / minimum speed moment inside corner
  if inCorneringSequence and (not state.apex_latched) and (curEv.duration or 0) >= 0.25 then
    local isYawPeak = (f.yaw_rate > 14.0 and f.yaw_accel <= 3.0)
    local isSpeedMin = (f.accel >= -1.0 and f.accel <= 0.8 and f.yaw_rate > 13.0)
    if (isYawPeak or isSpeedMin) and prevType ~= 'APEX' and prevType ~= 'EXIT_CORNER' then
      local conf = 0.84
      if isYawPeak and isSpeedMin then conf = 0.92 end
      proposals['APEX'] = conf
    end
  end

  -- 5. CORNER / MID_CORNER: Sustained turning rate
  if (f.yaw_rate > 13.0 or (f.curv_current > 0.035 and f.yaw_rate > 8.5)) and f.accel < 1.6 then
    local conf = math.min(0.95, 0.68 + (f.yaw_rate - 13.0) * 0.012)
    if inCorneringSequence then conf = conf + 0.05 end
    proposals['CORNER'] = math.min(0.96, conf)
  end

  -- 6. EXIT_CORNER: Yaw rate declining out of corner
  if (prevType == 'APEX' or prevType == 'CORNER') and f.yaw_rate < 12.0 and f.yaw_accel < 0.0 and f.accel > -1.0 then
    local conf = math.min(0.88, 0.72 + (12.0 - f.yaw_rate) * 0.015)
    proposals['EXIT_CORNER'] = conf
  end

  -- 7. APPROACH_CORNER: Straight road with upcoming sharp curvature ahead
  if f.curv_ahead > 0.045 and f.curv_current < 0.025 and f.yaw_rate < 7.5 and f.speed_kmh > 45.0 and f.accel > -2.0 then
    local conf = math.min(0.86, 0.65 + (f.curv_ahead - 0.045) * 3.5)
    proposals['APPROACH_CORNER'] = conf
  end

  -- 8. CRUISE / STRAIGHT: Low turning, stable speed
  if f.yaw_rate < 7.0 and math.abs(f.accel) < 1.1 and f.curv_ahead < 0.035 then
    local conf = math.min(0.95, 0.82 + (7.0 - f.yaw_rate) * 0.018)
    proposals['CRUISE'] = conf
  end

  -- Select proposal with highest confidence
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
      bestConf = 0.62
    elseif f.accel > 0.8 then
      bestType = 'ACCELERATION'
      bestConf = 0.64
    elseif f.accel < -1.5 then
      bestType = 'BRAKING'
      bestConf = 0.64
    else
      bestType = 'CRUISE'
      bestConf = 0.72
    end
  end

  return bestType, math.floor(bestConf * 100) / 100
end

-- ============================================================================
-- Event Coalescing with Macro Sequence & Apex Latch
-- ============================================================================

local function updateDrivingPhase(f)
  local curEv = state.driving_event
  local curDuration = f.t - curEv.start_time

  -- 1. Macro Sequence Transition Logic
  if state.macro_state == 'STRAIGHT_RUN' then
    if f.accel < -2.2 or f.yaw_rate > 11.0 or f.curv_ahead > 0.045 then
      state.macro_state = 'CORNER_SEQUENCE'
      state.apex_latched = false -- Ready for fresh apex in this corner!
      ac.log(string.format('[DirectorIntegration] Entering CORNER_SEQUENCE at %.2fs', f.t))
    end
  elseif state.macro_state == 'CORNER_SEQUENCE' then
    -- Exit corner sequence once car is accelerating straight on open road
    if f.accel > 1.0 and f.yaw_rate < 8.0 and f.curv_current < 0.025 and curDuration >= 0.6 then
      state.macro_state = 'STRAIGHT_RUN'
      state.apex_latched = false
      ac.log(string.format('[DirectorIntegration] Exiting to STRAIGHT_RUN at %.2fs', f.t))
    end
  end

  -- 2. Multi-Proposal Rule Evaluation
  local proposedState, proposedConf = evaluateEventRules(f, curEv)
  if proposedState == 'APEX' and state.apex_latched then
    proposedState = 'CORNER'
  end

  local importance = calculateEventImportance(proposedState, f)

  -- 3. Debounce State Machine (proven from event_detection_spike)
  if proposedState == curEv.event_type then
    state.pending_state = nil
    state.pending_frames = 0
    curEv.duration = math.floor(curDuration * 100) / 100
    curEv.confidence = math.floor((0.92 * curEv.confidence + 0.08 * proposedConf) * 100) / 100
    curEv.importance = math.floor((0.90 * curEv.importance + 0.10 * importance) * 100) / 100
  else
    if proposedState == state.pending_state then
      state.pending_frames = state.pending_frames + 1
    else
      state.pending_state = proposedState
      state.pending_frames = 1
    end

    local reqFrames = DEFAULT_MIN_CONFIRM_FRAMES
    local minDur = (curEv.event_type == 'APEX') and APEX_MIN_DURATION_S or DEFAULT_MIN_DURATION_S

    -- Emergency fast override for heavy braking
    if proposedState == 'BRAKING' and f.accel < -4.0 then
      reqFrames = FAST_CONFIRM_FRAMES
      minDur = 0.15
    end

    if state.pending_frames >= reqFrames and curDuration >= minDur then
      if proposedState == 'APEX' then
        state.apex_latched = true
      end

      state.driving_event = {
        event_type = proposedState,
        confidence = proposedConf,
        importance = importance,
        start_time = f.t,
        start_frame = f.frame,
        duration = 0.0,
      }
      state.pending_state = nil
      state.pending_frames = 0
      ac.log(string.format('[DirectorIntegration] Event -> %-15s (imp: %.2f, conf: %.2f) at %.2fs',
        proposedState, importance, proposedConf, f.t))
    else
      curEv.duration = math.floor(curDuration * 100) / 100
    end
  end
end

-- ============================================================================
-- Camera Style Policy: Desired Shot Profile per Event
-- ============================================================================

local function getCameraStylePolicy(eventType, importance)
  local policy = {
    angle_weights = { oncoming = 10, side_pan = 10, receding = 4 },
    category_weights = { elevated = 10, roadside = 10, high = 6, low = 4, wide = 2 },
    preferred_dist_min = 12.0,
    preferred_dist_max = 28.0,
    target_duration = PREFERRED_SHOT_DURATION,
  }

  if eventType == 'BRAKING' then
    -- Show car decelerating into camera: medium/wide, preferred oncoming, elevated
    policy.angle_weights = { oncoming = 16, side_pan = 6, receding = -8 }
    policy.category_weights = { elevated = 16, high = 12, roadside = 8, low = 2, wide = 3 }
    policy.preferred_dist_min = 18.0
    policy.preferred_dist_max = 34.0
    policy.target_duration = 3.2

  elseif eventType == 'TURN_IN' then
    -- Dynamic steering pitch: side_pan / elevated / roadside
    policy.angle_weights = { side_pan = 14, oncoming = 10, receding = -4 }
    policy.category_weights = { elevated = 16, roadside = 9, low = 3, high = 10, wide = 2 }
    policy.preferred_dist_min = 14.0
    policy.preferred_dist_max = 26.0
    policy.target_duration = 2.8

  elseif eventType == 'CORNER' then
    -- Show curvature arc and drift angle: side_pan / elevated / high
    policy.angle_weights = { side_pan = 15, oncoming = 9, receding = 6 }
    policy.category_weights = { elevated = 18, high = 14, roadside = 8, low = 2, wide = 4 }
    policy.preferred_dist_min = 16.0
    policy.preferred_dist_max = 34.0
    policy.target_duration = 3.8

  elseif eventType == 'APEX' then
    -- Dramatic low curb shot clipping apex!
    policy.angle_weights = { side_pan = 15, oncoming = 11, receding = -4 }
    policy.category_weights = { low = 18, roadside = 11, elevated = 6, high = 0, wide = 0 }
    policy.preferred_dist_min = 6.0
    policy.preferred_dist_max = 18.0
    policy.target_duration = 1.8

  elseif eventType == 'EXIT_CORNER' then
    -- Powering past camera: side / receding
    policy.angle_weights = { side_pan = 11, receding = 12, oncoming = 4 }
    policy.category_weights = { roadside = 12, elevated = 11, high = 6, low = 4, wide = 3 }
    policy.preferred_dist_min = 12.0
    policy.preferred_dist_max = 25.0
    policy.target_duration = 2.6

  elseif eventType == 'ACCELERATION' then
    -- Pulling away on straight: receding / fly-by / high crane
    policy.angle_weights = { receding = 16, side_pan = 10, oncoming = 10 }
    policy.category_weights = { high = 14, elevated = 12, roadside = 9, low = 2, wide = 4 }
    policy.preferred_dist_min = 18.0
    policy.preferred_dist_max = 42.0
    policy.target_duration = 3.6

  elseif eventType == 'APPROACH_CORNER' then
    policy.angle_weights = { oncoming = 15, side_pan = 8, receding = 0 }
    policy.category_weights = { high = 14, elevated = 12, roadside = 8, low = 2, wide = 3 }
    policy.preferred_dist_min = 20.0
    policy.preferred_dist_max = 35.0
    policy.target_duration = 3.0

  elseif eventType == 'CRUISE' then
    policy.angle_weights = { oncoming = 12, side_pan = 10, receding = 8 }
    policy.category_weights = { high = 12, elevated = 12, roadside = 9, low = 3, wide = 5 }
    policy.preferred_dist_min = 18.0
    policy.preferred_dist_max = 42.0
    policy.target_duration = 4.2
  end

  return policy
end

-- ============================================================================
-- Candidate Generation at Station
-- ============================================================================

local function generateCandidates(carPos, carProgress)
  local accepted = {}

  for _, progOffset in ipairs(PROGRESS_OFFSETS) do
    local p = carProgress + progOffset
    if p < 0 then p = p + 1.0 end
    if p > 1 then p = p - 1.0 end

    for _, camType in ipairs(CAMERA_TYPES) do
      state.eval_count = state.eval_count + 1

      local twOk, worldPos = pcall(function()
        return ac.trackCoordinateToWorld(vec3(camType.lateral, 0, p))
      end)
      if not twOk or worldPos == nil then
        state.eval_rejected = state.eval_rejected + 1
        goto continue_cam
      end

      local groundY, hitDist = findGround(worldPos)
      if groundY == nil then
        state.eval_rejected = state.eval_rejected + 1
        goto continue_cam
      end

      local camY = groundY + camType.height
      local camPos = vec3(worldPos.x, camY, worldPos.z)

      if camY < groundY + MIN_CLEARANCE then
        state.eval_rejected = state.eval_rejected + 1
        goto continue_cam
      end

      local dist = v3dist(camPos, carPos)
      local maxInitDist = (state.macro_state == 'STRAIGHT_RUN') and 45.0 or 38.0
      if dist < 2.5 or dist > maxInitDist then
        state.eval_rejected = state.eval_rejected + 1
        goto continue_cam
      end

      local target = vec3(carPos.x, carPos.y + 0.6, carPos.z)
      local losClear = checkLineOfSight(camPos, target)
      if not losClear then
        state.eval_rejected = state.eval_rejected + 1
        goto continue_cam
      end

      state.eval_accepted = state.eval_accepted + 1
      accepted[#accepted + 1] = {
        type = camType.name,
        category = camType.category,
        side = camType.side,
        desc = camType.desc,
        lateral = camType.lateral,
        height = camType.height,
        progress = p,
        prog_offset = progOffset,
        final_pos = camPos,
        ground_y = groundY,
        distance_to_car = dist,
      }

      ::continue_cam::
    end
  end

  return accepted
end

-- ============================================================================
-- Multi-Criteria Camera Scoring Engine
-- ============================================================================

local function scoreCandidate(c, currentShot, carPos, carLook, eventType, importance, policy)
  -- 1. Geometric Angle Classification
  local toCam = v3norm(vec3(c.final_pos.x - carPos.x, 0, c.final_pos.z - carPos.z))
  local flatLook = v3norm(vec3(carLook.x, 0, carLook.z))
  local dotDir = v3dot(flatLook, toCam)

  local angleType = 'side_pan'
  if dotDir > 0.25 then
    angleType = 'oncoming'
  elseif dotDir < -0.25 then
    angleType = 'receding'
  else
    angleType = 'side_pan'
  end

  -- 2. Placement Base Score & Distance Fit
  local placement_score = 25.0
  local d = c.distance_to_car
  if d >= 12.0 and d <= 32.0 then
    placement_score = placement_score + 18.0
  elseif d >= 6.0 and d < 12.0 then
    placement_score = placement_score + 14.0
  elseif d > 32.0 and d <= 42.0 then
    placement_score = placement_score + 12.0
  else
    placement_score = placement_score + 4.0
  end

  -- 3. Event Style Policy Match (Weighted by Event Importance)
  local angleBonus = policy.angle_weights[angleType] or 5.0
  local catBonus = policy.category_weights[c.category] or 5.0
  local distBonus = (d >= policy.preferred_dist_min and d <= policy.preferred_dist_max) and 6.0 or 0.0

  local event_match = (angleBonus + catBonus + distBonus) * importance

  -- 4. Visual Quality (Elevated boom / crane over bushes)
  local visual_quality = 0.0
  if c.category == 'elevated' then
    visual_quality = 12.0
  elseif c.category == 'roadside' then
    visual_quality = 10.0
  elseif c.category == 'low' then
    visual_quality = 10.0
  elseif c.category == 'high' then
    visual_quality = 8.0
  elseif c.category == 'wide' then
    visual_quality = 5.0
  end

  -- 5. Soft Diversity Rules (Avoid Repetitive Shots)
  local diversity = 0.0
  if currentShot ~= nil then
    -- Hard penalty against identical camera
    if c.type == currentShot.camera_type then
      diversity = diversity - 26.0
    end
    -- Penalty against same category
    if c.category == currentShot.category then
      diversity = diversity - 8.0
    else
      diversity = diversity + 6.0
    end
    -- Side alternation preference
    if c.side ~= 'center' and currentShot.side ~= 'center' then
      if c.side == currentShot.side then
        diversity = diversity - 4.0
      else
        diversity = diversity + 5.0
      end
    end
    -- Angle alternation
    if currentShot.angle_type == angleType then
      diversity = diversity - 4.0
    else
      diversity = diversity + 6.0
    end
  end

  -- 6. Pacing Modulation
  local pacing = 0.0

  local raw = placement_score + event_match + visual_quality + diversity + pacing
  local finalScore = math.max(10.0, math.min(100.0, math.floor(raw * 10 + 0.5) / 10))

  c.calculated_score = finalScore
  c.angle_type = angleType
  c.score_components = {
    placement = math.floor(placement_score * 10) / 10,
    event_match = math.floor(event_match * 10) / 10,
    visual_quality = math.floor(visual_quality * 10) / 10,
    diversity = math.floor(diversity * 10) / 10,
    pacing = math.floor(pacing * 10) / 10,
    raw = math.floor(raw * 10 + 0.5) / 10,
    final = finalScore,
    dot_dir = math.floor(dotDir * 1000) / 1000,
  }

  return finalScore
end

-- ============================================================================
-- Pacing & Cut Decision Logic
-- ============================================================================

local function evaluateCutDecision(currentFrame, replayTime, currentShot, drivingEvent, candidates)
  if currentShot == nil then
    return true, 'INIT'
  end

  local timeInShot = replayTime - currentShot.start_time

  -- Determine state-dependent envelopes
  local isStraightState = (state.macro_state == 'STRAIGHT_RUN') or
                          (drivingEvent.event_type == 'ACCELERATION') or
                          (drivingEvent.event_type == 'CRUISE')

  local cooldownSec = isStraightState and 2.0 or 1.6

  local camPos = vec3(currentShot.camera_position[1], currentShot.camera_position[2], currentShot.camera_position[3])
  local carPos = state.samples[#state.samples].pos
  local dist = v3dist(camPos, carPos)
  local losOk = checkLineOfSight(camPos, vec3(carPos.x, carPos.y + 0.6, carPos.z))

  -- Track consecutive occluded frames to filter 1-frame light pole / fence / brush pass
  if not losOk then
    state.los_occluded_frames = (state.los_occluded_frames or 0) + 1
  else
    state.los_occluded_frames = 0
  end

  -- Determine distance boundaries based on state & angle
  local maxAllowedDist = 45.0
  if isStraightState then
    if currentShot.category == 'high' then
      maxAllowedDist = 80.0
    elseif currentShot.angle_type == 'receding' then
      maxAllowedDist = 72.0
    else
      maxAllowedDist = 65.0
    end
  else
    if currentShot.category == 'high' then
      maxAllowedDist = 60.0
    elseif currentShot.category == 'elevated' then
      maxAllowedDist = 48.0
    else
      maxAllowedDist = 42.0
    end
  end

  -- Emergency Cooldown: protect shot from premature cutting
  if timeInShot < cooldownSec then
    -- Hard safety cut only if car is dangerously close, completely out of sight (>85m), or blocked for >= 6 frames (~0.18s)
    if dist < 1.8 or dist > 85.0 or state.los_occluded_frames >= 6 then
      return true, 'VISIBILITY_LOST'
    end
    return false, nil
  end

  -- Reason A: Visibility Lost / Out of distance range
  local occlusionThreshold = (state.macro_state == 'CORNER_SEQUENCE') and 5 or 4
  if dist > maxAllowedDist or dist < 2.5 or state.los_occluded_frames >= occlusionThreshold then
    return true, 'VISIBILITY_LOST'
  end

  -- Reason B: Significant Driving Event Change
  if drivingEvent.event_type ~= currentShot.event_type then
    local allowEventCut = false

    if state.macro_state == 'STRAIGHT_RUN' then
      local minShotDur = isStraightState and 2.5 or MIN_SHOT_DURATION_DEFAULT
      if timeInShot >= minShotDur and drivingEvent.importance >= 0.60 and drivingEvent.duration >= 0.25 then
        allowEventCut = true
      end
    else
      -- Inside CORNER_SEQUENCE:
      -- Suppress cuts caused by pedal modulations (gas tap / trail brake / slide adjustments)!
      local isIntraCornerPedalFluctuation =
        (currentShot.event_type == 'CORNER' and (drivingEvent.event_type == 'ACCELERATION' or drivingEvent.event_type == 'BRAKING')) or
        (currentShot.event_type == 'BRAKING' and drivingEvent.event_type == 'CORNER' and timeInShot < 2.8) or
        (currentShot.event_type == 'ACCELERATION' and drivingEvent.event_type == 'CORNER')

      if not isIntraCornerPedalFluctuation then
        -- Allow cut for APEX (dramatic curb shot) or deliberate exit cut
        local minShotDur = (currentShot.event_type == 'APEX') and MIN_SHOT_DURATION_APEX or 3.0
        if timeInShot >= minShotDur and drivingEvent.importance >= 0.65 and drivingEvent.duration >= 0.30 then
          allowEventCut = true
        end
      end
    end

    if allowEventCut then
      return true, 'EVENT_CHANGE'
    end
  end

  -- Reason C: Max Duration Exceeded
  local maxDur = isStraightState and MAX_SHOT_DURATION_CRUISE or MAX_SHOT_DURATION_DEFAULT
  if timeInShot >= maxDur then
    return true, 'MAX_DURATION'
  end

  -- Reason D: Significantly Better Candidate Appears
  local preferredDur = isStraightState and 3.2 or 3.2
  if timeInShot >= preferredDur and candidates and #candidates > 0 then
    local bestCand = candidates[1]
    if bestCand and (bestCand.calculated_score or 0) > (currentShot.final_score + SCORE_IMPROVEMENT_THRESHOLD) then
      return true, 'BETTER_CANDIDATE'
    end
  end

  -- Reason E: Corner Sequence Ended
  if state.macro_state == 'STRAIGHT_RUN' and currentShot.macro_state == 'CORNER_SEQUENCE' and timeInShot >= 2.2 then
    return true, 'SEQUENCE_END'
  end

  return false, nil
end

-- ============================================================================
-- Apply Camera & Execute Cut
-- ============================================================================

local function executeCut(best, cutReason, currentFrame, replayTime, carPos, drivingEvent)
  -- 1. Finalize previous shot
  if state.current_shot ~= nil and #state.session_shots > 0 then
    local prevShot = state.session_shots[#state.session_shots]
    local actualEndFrame = currentFrame - 1
    actualEndFrame = math.max(prevShot.start_frame, actualEndFrame)
    prevShot.end_frame = actualEndFrame
    prevShot.end_time = math.floor((actualEndFrame / state.sample_rate) * 100) / 100
    local fCount = actualEndFrame - prevShot.start_frame + 1
    prevShot.duration = math.floor((fCount / state.sample_rate) * 100) / 100
  end

  -- 2. Create new shot record
  local shotRecord = {
    index = #state.session_shots + 1,
    start_frame = currentFrame,
    end_frame = currentFrame,
    start_time = math.floor(replayTime * 100) / 100,
    end_time = math.floor(replayTime * 100) / 100,
    duration = 0.0,
    event_type = drivingEvent.event_type,
    event_confidence = drivingEvent.confidence,
    event_importance = drivingEvent.importance,
    macro_state = state.macro_state,
    camera_type = best.type,
    category = best.category,
    side = best.side,
    camera_position = {
      math.floor(best.final_pos.x * 100) / 100,
      math.floor(best.final_pos.y * 100) / 100,
      math.floor(best.final_pos.z * 100) / 100,
    },
    distance = math.floor(best.distance_to_car * 10) / 10,
    angle_type = best.angle_type,
    final_score = best.calculated_score,
    score_components = best.score_components,
    cut_reason = cutReason,
  }

  state.session_shots[#state.session_shots + 1] = shotRecord
  state.current_shot = shotRecord
  state.shot_start_frame = currentFrame
  state.shot_start_time = replayTime
  state.last_cut_reason = cutReason
  state.los_occluded_frames = 0

  ac.log(string.format('[DirectorIntegration] CUT #%02d [%s] -> %s (%s, dist=%.1fm, score=%.1f) Event: %s (imp:%.2f)',
    shotRecord.index, cutReason, best.type, best.angle_type, best.distance_to_car, best.calculated_score,
    drivingEvent.event_type, drivingEvent.importance))
end

-- ============================================================================
-- Main Director Update Loop
-- ============================================================================

local function updateDirector()
  if not state.director_active then return end

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
  if state.last_processed_frame ~= nil then
    if math.abs(currentFrame - state.last_processed_frame) > 40 then
      ac.log(string.format('[DirectorIntegration] Scrub detected (%d -> %d). Re-syncing.',
        state.last_processed_frame, currentFrame))
      state.samples = {}
      state.current_shot = nil
    end
  end

  if isPaused or currentFrame == state.last_processed_frame then
    -- Keep aiming camera if paused
    goto apply_camera_transform
  end

  state.last_processed_frame = currentFrame

  -- 1. Ingest telemetry sample
  do
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
    while #state.samples > state.max_samples do
      table.remove(state.samples, 1)
    end
  end

  -- 2. Compute kinematics & update driving phase
  do
    local k = computeKinematics()
    if k ~= nil then
      updateDrivingPhase(k)
    end
  end

  -- 3. Evaluate candidate cameras with Event Style Policy
  do
    local replayTime = currentFrame / state.sample_rate
    local candidates = generateCandidates(carPos, carProgress)
    local policy = getCameraStylePolicy(state.driving_event.event_type, state.driving_event.importance)

    if #candidates > 0 then
      for _, cand in ipairs(candidates) do
        scoreCandidate(cand, state.current_shot, carPos, carLook, state.driving_event.event_type, state.driving_event.importance, policy)
      end
      table.sort(candidates, function(a, b) return (a.calculated_score or 0) > (b.calculated_score or 0) end)
    end

    -- 4. Check if cut should trigger
    local shouldCut, cutReason = evaluateCutDecision(currentFrame, replayTime, state.current_shot, state.driving_event, candidates)

    if shouldCut and #candidates > 0 then
      local best = candidates[1]
      executeCut(best, cutReason, currentFrame, replayTime, carPos, state.driving_event)
    end
  end

  ::apply_camera_transform::

  -- 5. Per-frame camera aim at car
  if state.director_cam and state.current_shot then
    local target = vec3(carPos.x, carPos.y + 0.6, carPos.z)
    local camPos = vec3(state.current_shot.camera_position[1],
                        state.current_shot.camera_position[2],
                        state.current_shot.camera_position[3])

    local look = v3norm(vec3(target.x - camPos.x, target.y - camPos.y, target.z - camPos.z))
    local worldUp = vec3(0, 1, 0)
    if math.abs(v3dot(look, worldUp)) > 0.999 then worldUp = vec3(0, 0, 1) end
    local right = v3norm(v3cross(look, worldUp))
    local up = v3cross(right, look)

    state.director_cam.transform.position = camPos
    state.director_cam.transform.look = look
    state.director_cam.transform.side = right
    state.director_cam.transform.up = up
    state.director_cam.fov = 55
  end
end

-- ============================================================================
-- Start / Stop Director
-- ============================================================================

local function startDirector()
  if state.director_active then return end

  local ok, cam = pcall(function()
    return ac.grabCamera('Event Aware Director')
  end)
  if not ok or cam == nil then
    state.status_message = 'Failed to grab camera: ' .. tostring(cam)
    return
  end

  state.director_cam = cam
  state.director_active = true
  local sim = ac.getSim()
  state.session_start_frame = sim and sim.replayCurrentFrame or 0
  state.last_processed_frame = nil
  state.samples = {}
  state.session_shots = {}
  state.current_shot = nil
  state.macro_state = 'STRAIGHT_RUN'
  state.apex_latched = false
  local curT = (state.session_start_frame or 0) / state.sample_rate
  state.driving_event = {
    event_type = 'CRUISE',
    confidence = 0.90,
    importance = 0.50,
    start_time = curT,
    start_frame = state.session_start_frame,
    duration = 0,
  }
  state.pending_state = nil
  state.pending_frames = 0
  state.los_occluded_frames = 0
  state.eval_count = 0
  state.eval_accepted = 0
  state.eval_rejected = 0
  state.save_message = nil
  state.status_message = 'Event-Aware Director: ACTIVE'
  ac.log('[DirectorIntegration] Started Event-Aware Director at frame ' .. state.session_start_frame)
end

local function stopDirector(andSave)
  if not state.director_active then return end

  local sim = ac.getSim()
  local currentFrame = sim and sim.replayCurrentFrame or (state.last_processed_frame or 0)

  -- Finalize last shot
  if state.current_shot and #state.session_shots > 0 then
    local lastShot = state.session_shots[#state.session_shots]
    local actualEnd = math.max(lastShot.start_frame, currentFrame)
    lastShot.end_frame = actualEnd
    lastShot.end_time = math.floor((actualEnd / state.sample_rate) * 100) / 100
    local fCount = actualEnd - lastShot.start_frame + 1
    lastShot.duration = math.floor((fCount / state.sample_rate) * 100) / 100
  end

  if state.director_cam then
    pcall(function() state.director_cam:dispose() end)
    state.director_cam = nil
  end

  state.director_active = false
  state.status_message = string.format('Director stopped. %d shots recorded.', #state.session_shots)
  ac.log('[DirectorIntegration] Stopped. Total shots: ' .. #state.session_shots)
end

-- ============================================================================
-- Save Director Integration Report JSON
-- ============================================================================

local function saveDirectorReport()
  if #state.session_shots == 0 then
    state.save_message = 'Error: no recorded shots to save. Run replay test first.'
    return false
  end

  local docsOk, docs = pcall(function() return ac.getFolder(ac.FolderID.ACDocuments) end)
  if not docsOk or docs == nil then
    state.save_message = 'Error: cannot locate Documents folder'
    return false
  end

  local path = docs .. '/director_integration_report.json'

  -- Calculate summary metrics
  local startF = state.session_shots[1].start_frame
  local endF = state.session_shots[#state.session_shots].end_frame
  local totalDur = math.floor(((endF - startF + 1) / state.sample_rate) * 100) / 100

  local typeDist = {}
  local eventDist = {}
  local cutReasons = {}
  local sumImp = 0
  local sumConf = 0
  local sumDur = 0

  for _, s in ipairs(state.session_shots) do
    typeDist[s.camera_type] = (typeDist[s.camera_type] or 0) + 1
    eventDist[s.event_type] = (eventDist[s.event_type] or 0) + 1
    cutReasons[s.cut_reason] = (cutReasons[s.cut_reason] or 0) + 1
    sumImp = sumImp + (s.event_importance or 0)
    sumConf = sumConf + (s.event_confidence or 0)
    sumDur = sumDur + (s.duration or 0)
  end

  local shotCount = #state.session_shots
  local avgDur = shotCount > 0 and (math.floor((sumDur / shotCount) * 100) / 100) or 0
  local avgImp = shotCount > 0 and (math.floor((sumImp / shotCount) * 100) / 100) or 0
  local avgConf = shotCount > 0 and (math.floor((sumConf / shotCount) * 100) / 100) or 0
  local shotsPerMin = totalDur > 0 and (math.floor((shotCount / (totalDur / 60.0)) * 10) / 10) or 0

  local report = {
    schema_version = 1,
    app = 'director_integration_spike',
    csp_version = state.csp,
    timestamp = os.time(),
    sample_rate_hz = state.sample_rate,
    track_length_m = state.track_length_m,
    summary = {
      total_replay_time_analysed = totalDur,
      session_start_frame = startF,
      session_end_frame = endF,
      total_shots = shotCount,
      average_shot_duration = avgDur,
      shots_per_minute = shotsPerMin,
      camera_type_distribution = typeDist,
      event_distribution = eventDist,
      cut_reasons = cutReasons,
      average_importance = avgImp,
      average_confidence = avgConf,
      candidate_evaluations = {
        total = state.eval_count,
        accepted = state.eval_accepted,
        rejected = state.eval_rejected,
        acceptance_pct = state.eval_count > 0 and (math.floor((state.eval_accepted / state.eval_count) * 1000) / 10) or 0,
      },
    },
    shots = state.session_shots,
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

  state.save_message = string.format('Report saved: %s (%d shots, %.1fs)', path, shotCount, totalDur)
  ac.log('[DirectorIntegration] Saved report: ' .. path)
  return true
end

-- ============================================================================
-- UI Window
-- ============================================================================

local function drawWindow()
  ui.text('=== EVENT-AWARE CAMERA DIRECTOR SPIKE ===')
  ui.text('CSP: ' .. state.csp)

  -- Telemetry Header
  local sim = ac.getSim()
  local currentFrame = sim and sim.replayCurrentFrame or 0
  local replayTime = currentFrame / state.sample_rate

  ui.text(string.format('Replay: frame %d | t = %.2fs | rate = %.1fx',
    currentFrame, replayTime, sim and sim.replayPlaybackRate or 1.0))
  ui.text(string.format('Car: Spd %.1f km/h | Progress: %.4f',
    state.live.speed_kmh, state.live.progress))

  ui.separator()

  -- Action Controls
  if not state.director_active then
    if ui.button('START EVENT-AWARE DIRECTOR', vec2(250, 34)) then
      startDirector()
    end
    ui.sameLine()
    if #state.session_shots > 0 then
      if ui.button('Save Director Session', vec2(170, 34)) then
        saveDirectorReport()
      end
      ui.sameLine()
      if ui.button('Clear', vec2(70, 34)) then
        state.session_shots = {}
        state.current_shot = nil
        state.save_message = nil
        state.status_message = 'Cleared session buffer.'
      end
    end
  else
    if ui.button('STOP DIRECTOR', vec2(200, 34)) then
      stopDirector()
    end
    ui.sameLine()
    if #state.session_shots > 0 then
      if ui.button('Save Session', vec2(140, 34)) then
        saveDirectorReport()
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

  -- Live Status Card
  if state.director_active and state.current_shot then
    local s = state.current_shot
    local ev = state.driving_event
    local timeInShot = replayTime - state.shot_start_time

    ui.text('=== CURRENT ACTIVE SHOT ===')
    ui.text(string.format('EVENT:      %-15s (Conf: %.2f | Imp: %.2f)',
      ev.event_type, ev.confidence, ev.importance))
    ui.text(string.format('MACRO:      %-15s (Apex Latched: %s)',
      state.macro_state, state.apex_latched and 'YES' or 'NO'))
    ui.text(string.format('CAMERA:     %s (%s, side=%s) | Dist: %.1fm',
      s.camera_type, s.angle_type, s.side, s.distance))
    ui.text(string.format('TIMING:     %.2fs in shot (Preferred: %.1fs) | Score: %.1f',
      timeInShot, PREFERRED_SHOT_DURATION, s.final_score))
    ui.text(string.format('CUT REASON: [%s] (Last frame: %d)',
      state.last_cut_reason, state.shot_start_frame))
  elseif not state.director_active and #state.session_shots > 0 then
    ui.text('Director idle. Click START EVENT-AWARE DIRECTOR to record.')
  else
    ui.text('Press START EVENT-AWARE DIRECTOR to begin live directing.')
  end

  -- Summary Statistics
  if #state.session_shots > 0 then
    ui.separator()
    ui.text(string.format('=== RECORDED SESSION SUMMARY (%d shots) ===', #state.session_shots))

    local totalDur = 0
    local reasons = {}
    local camDist = {}
    local evDist = {}

    for _, shot in ipairs(state.session_shots) do
      totalDur = totalDur + (shot.duration or 0)
      reasons[shot.cut_reason] = (reasons[shot.cut_reason] or 0) + 1
      camDist[shot.camera_type] = (camDist[shot.camera_type] or 0) + 1
      evDist[shot.event_type] = (evDist[shot.event_type] or 0) + 1
    end

    local avgDur = #state.session_shots > 0 and (totalDur / #state.session_shots) or 0
    local spm = totalDur > 0 and (#state.session_shots / (totalDur / 60.0)) or 0
    ui.text(string.format('Total duration: %.1fs | Avg shot: %.2fs | Pacing: %.1f shots/min',
      totalDur, avgDur, spm))

    ui.separator()
    ui.text('--- Cut Reasons Breakdown ---')
    local reasonStr = {}
    for r, cnt in pairs(reasons) do
      table.insert(reasonStr, string.format('%s: %d', r, cnt))
    end
    ui.textWrapped(table.concat(reasonStr, ' | '))

    ui.separator()
    ui.text('--- Camera Type Distribution ---')
    for tName, cnt in pairs(camDist) do
      local pct = (cnt / #state.session_shots) * 100
      ui.text(string.format('  %-22s: %d  (%.1f%%)', tName, cnt, pct))
    end

    -- Recent Decisions Log
    ui.separator()
    ui.text('=== RECENT DIRECTORIAL DECISIONS ===')
    local startIdx = math.max(1, #state.session_shots - 9)
    for i = #state.session_shots, startIdx, -1 do
      local sh = state.session_shots[i]
      ui.text(string.format('%02d. %05.1fs  [%-13s] %-12s -> %-20s (%.2fs, sc:%.0f)',
        sh.index, sh.start_time, sh.cut_reason, sh.event_type, sh.camera_type, sh.duration, sh.final_score))
    end
  end

  -- Telemetry & Style Policy Diagnostics Panel
  ui.separator()
  local diagLabel = state.show_diagnostics and '[-] Hide Diagnostics Panel' or '[+] Show Diagnostics Panel'
  if ui.button(diagLabel, vec2(260, 24)) then
    state.show_diagnostics = not state.show_diagnostics
  end

  if state.show_diagnostics then
    ui.text('--- Live Kinematic Derivatives ---')
    ui.text(string.format('Accel:  %+.2f m/s² (%+.2f G) | Yaw: %.1f °/s (dω/dt: %+.1f)',
      state.live.accel_ms2, state.live.accel_g, state.live.yaw_rate_dps, state.live.yaw_accel_dps2))
    ui.text(string.format('Curv:   current %.3f rad | ahead: %.3f rad | Lat: %+.2fm',
      state.live.curv_current, state.live.curv_ahead, state.live.lateral_m))

    if state.current_shot and state.current_shot.score_components then
      local sc = state.current_shot.score_components
      ui.text('--- Current Shot Score Breakdown ---')
      ui.text(string.format('Placement: %.1f | EventMatch: %.1f | VisQuality: %.1f',
        sc.placement, sc.event_match, sc.visual_quality))
      ui.text(string.format('Diversity: %+.1f | Pacing: %+.1f | FINAL: %.1f',
        sc.diversity, sc.pacing, sc.final))
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
  end

  updateDirector()
  drawWindow()
end
