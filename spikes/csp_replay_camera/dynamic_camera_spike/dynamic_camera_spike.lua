-- Dynamic Camera Spike — CSP 0.3.0-preview595
--
-- Procedural Dynamic Camera Motion System
-- Implements 6 active operator motion profiles:
--   1. DOLLY_IN   (Dynamic push-in with telephoto compression)
--   2. DOLLY_OUT  (Dynamic pull-out / crane elevation)
--   3. SIDE_TRACK (Parallel vehicle tracking with low-angle flank framing)
--   4. CHASE      (Dynamic follow cam with acceleration modulation & banking)
--   5. ORBIT      (Arcing panoramic sweep across vehicle turn angle)
--   6. FLY_BY     (Roadside anchor with subtle drift and whip-pan zoom)
--
-- Features:
--   - Per-frame interpolated positions, targets, and FOV curves
--   - Smooth easing (linear, easeInQuad, easeOutQuad, easeInOutQuad, easeInOutCubic)
--   - Dutch roll angle banking during turns (-2.5 deg to +2.5 deg)
--   - Obstacle raycast safety and ground clearance clamping
--   - Dual UI modes: Manual Profile Tester + Autonomous Telemetry Director
--   - JSON diagnostic report export (dynamic_camera_report.json)
--
-- Does NOT modify timeline_driver_spike, aidirector_spike, camera_placement_spike,
-- shot_planner_spike, event_detection_spike, or director_integration_spike.

local json = require('json')

-- ============================================================================
-- Configuration & Calibrated Parameters
-- ============================================================================

local RAISE_HEIGHT = 80.0
local RAY_LENGTH = 200.0
local MIN_CLEARANCE = 0.5
local LOS_TOLERANCE = 2.0

-- 10 Calibrated Camera Anchors
local CAMERA_TYPES = {
  { name = 'roadside_left_close',   lateral = -1.25, height = 1.4, category = 'roadside', side = 'left',   desc = 'Left edge, eye level' },
  { name = 'roadside_right_close',  lateral =  1.25, height = 1.4, category = 'roadside', side = 'right',  desc = 'Right edge, eye level' },
  { name = 'roadside_left_wide',    lateral = -1.65, height = 1.6, category = 'wide',     side = 'left',   desc = 'Wide left berm' },
  { name = 'roadside_right_wide',   lateral =  1.65, height = 1.6, category = 'wide',     side = 'right',  desc = 'Wide right berm' },
  { name = 'low_left',              lateral = -1.15, height = 0.65, category = 'low',     side = 'left',   desc = 'Low left curb view' },
  { name = 'low_right',             lateral =  1.15, height = 0.65, category = 'low',     side = 'right',  desc = 'Low right curb view' },
  { name = 'elevated_left_low',     lateral = -1.35, height = 2.8, category = 'elevated', side = 'left',   desc = 'Elevated left, over bushes (2.8m)' },
  { name = 'elevated_left_mid',     lateral = -1.45, height = 4.5, category = 'elevated', side = 'left',   desc = 'Elevated left boom pole (4.5m)' },
  { name = 'elevated_right',        lateral =  1.30, height = 3.5, category = 'elevated', side = 'right',  desc = 'Elevated right overview (3.5m)' },
  { name = 'elevated_high',         lateral = -0.80, height = 7.5, category = 'high',     side = 'center', desc = 'High crane / overhead (7.5m)' },
}

local PROGRESS_OFFSETS = { -0.0040, -0.0020, 0.0, 0.0020, 0.0040, 0.0055 }

-- Pacing & Cut Duration Thresholds
local MIN_SHOT_DURATION_DEFAULT = 1.8
local MIN_SHOT_DURATION_APEX = 1.2
local PREFERRED_SHOT_DURATION = 2.8
local MAX_SHOT_DURATION_DEFAULT = 4.2
local MAX_SHOT_DURATION_CRUISE = 5.2
local CUT_COOLDOWN_S = 1.0
local SCORE_IMPROVEMENT_THRESHOLD = 18.0

local DEFAULT_MIN_CONFIRM_FRAMES = 5
local FAST_CONFIRM_FRAMES = 2
local DEFAULT_MIN_DURATION_S = 0.35
local APEX_MIN_DURATION_S = 0.20

-- ============================================================================
-- State Machine
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

  -- UI mode: 1 = 'autonomous', 2 = 'manual'
  ui_mode = 1,

  -- Manual Profile Settings
  manual = {
    profile_type = 'SIDE_TRACK', -- DOLLY_IN, DOLLY_OUT, SIDE_TRACK, CHASE, ORBIT, FLY_BY
    intensity = 1.0,
    duration = 3.0,
    fov_start = 50.0,
    fov_end = 46.0,
    easing = 'easeInOutQuad',
    side = 'left', -- 'left' or 'right'
  },

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
  macro_state = 'STRAIGHT_RUN',
  apex_latched = false,

  -- Event Hysteresis & Visibility Stabilization
  pending_state = nil,
  pending_frames = 0,
  los_occluded_frames = 0,
  recent_cameras = {},

  -- Current Active Dynamic Shot & Motion Profile
  current_shot = nil,
  shot_start_frame = 0,
  shot_start_time = 0,
  last_cut_reason = 'INIT',

  -- Live Motion State
  live_motion = {
    profile_name = 'NONE',
    progress_norm = 0.0,
    eased_progress = 0.0,
    current_fov = 55.0,
    cam_dist = 0.0,
    cam_pos = nil,
    target_pos = nil,
    roll_deg = 0.0,
  },

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

local function v3lerp(a, b, t)
  return vec3(
    a.x + (b.x - a.x) * t,
    a.y + (b.y - a.y) * t,
    a.z + (b.z - a.z) * t
  )
end

local function clamp(val, minV, maxV)
  if val < minV then return minV end
  if val > maxV then return maxV end
  return val
end

-- ============================================================================
-- Easing Functions
-- ============================================================================

local function easeLinear(t)
  return t
end

local function easeInQuad(t)
  return t * t
end

local function easeOutQuad(t)
  return t * (2.0 - t)
end

local function easeInOutQuad(t)
  if t < 0.5 then
    return 2.0 * t * t
  else
    return -1.0 + (4.0 - 2.0 * t) * t
  end
end

local function easeOutCubic(t)
  local f = t - 1.0
  return f * f * f + 1.0
end

local function easeInOutCubic(t)
  if t < 0.5 then
    return 4.0 * t * t * t
  else
    local f = 2.0 * t - 2.0
    return 0.5 * f * f * f + 1.0
  end
end

local function applyEasing(tNorm, easingType)
  local t = clamp(tNorm, 0.0, 1.0)
  if easingType == 'easeOutQuad' then
    return easeOutQuad(t)
  elseif easingType == 'easeInOutQuad' then
    return easeInOutQuad(t)
  elseif easingType == 'easeInQuad' then
    return easeInQuad(t)
  elseif easingType == 'easeOutCubic' then
    return easeOutCubic(t)
  elseif easingType == 'easeInOutCubic' then
    return easeInOutCubic(t)
  else
    return easeLinear(t)
  end
end

-- ============================================================================
-- Raycasting & Terrain Probing
-- ============================================================================

local function findGround(pos)
  local rayStart = vec3(pos.x, pos.y + RAISE_HEIGHT, pos.z)
  local rayDir = vec3(0, -1, 0)
  local hitDist = physics.raycastTrack(rayStart, rayDir, RAY_LENGTH)
  if hitDist >= 0 and hitDist < RAY_LENGTH then
    return rayStart.y - hitDist, hitDist
  end
  return nil, nil
end

local function checkLineOfSight(camPos, targetPos)
  local toTarget = vec3(targetPos.x - camPos.x, targetPos.y - camPos.y, targetPos.z - camPos.z)
  local dist = v3len(toTarget)
  if dist < 0.001 then return true end
  local dir = vec3(toTarget.x / dist, toTarget.y / dist, toTarget.z / dist)
  local hit = physics.raycastTrack(camPos, dir, dist)
  if hit >= 0 and hit < (dist - LOS_TOLERANCE) then
    return false
  end
  return true
end

local function getTrackCurvature(progress, aheadOffset)
  local p1 = progress + (aheadOffset or 0.0)
  if p1 < 0 then p1 = p1 + 1.0 end
  if p1 > 1 then p1 = p1 - 1.0 end

  local p0 = p1 - 0.0015
  local p2 = p1 + 0.0015
  if p0 < 0 then p0 = p0 + 1.0 end
  if p2 > 1 then p2 = p2 - 1.0 end

  local ok0, w0 = pcall(function() return ac.trackCoordinateToWorld(vec3(0, 0, p0)) end)
  local ok1, w1 = pcall(function() return ac.trackCoordinateToWorld(vec3(0, 0, p1)) end)
  local ok2, w2 = pcall(function() return ac.trackCoordinateToWorld(vec3(0, 0, p2)) end)

  if not (ok0 and ok1 and ok2 and w0 and w1 and w2) then return 0.0 end

  local d1x, d1z = w1.x - w0.x, w1.z - w0.z
  local d2x, d2z = w2.x - w1.x, w2.z - w1.z

  local len1 = math.sqrt(d1x * d1x + d1z * d1z)
  local len2 = math.sqrt(d2x * d2x + d2z * d2z)
  if len1 < 0.01 or len2 < 0.01 then return 0.0 end

  d1x, d1z = d1x / len1, d1z / len1
  d2x, d2z = d2x / len2, d2z / len2

  local crossVal = d1x * d2z - d1z * d2x
  local dotVal = clamp(d1x * d2x + d1z * d2z, -1.0, 1.0)
  local angle = math.abs(math.atan2(crossVal, dotVal))
  local segLen = (len1 + len2) * 0.5
  if segLen < 0.01 then return 0.0 end

  return angle / segLen
end

-- ============================================================================
-- Telemetry Kinematics Calculation
-- ============================================================================

local function computeKinematics()
  local n = #state.samples
  if n < 4 then return nil end

  local s_now = state.samples[n]
  local s_short = state.samples[n - 1]
  local s_mid = (n >= 4) and state.samples[n - 3] or state.samples[1]

  local dt_short = s_now.t - s_short.t
  local dt_mid = s_now.t - s_mid.t
  if dt_short <= 0.0001 then dt_short = 0.03 end
  if dt_mid <= 0.0001 then dt_mid = 0.09 end

  local a_inst = (s_now.speed_ms - s_short.speed_ms) / dt_short
  local a_mid = (s_now.speed_ms - s_mid.speed_ms) / dt_mid
  local a_filtered = a_inst * 0.45 + a_mid * 0.55
  local a_g = a_filtered / 9.80665

  local yaw_rate = 0
  local l_now = s_now.look
  local l_short = s_short.look
  local dotP = clamp(l_now.x * l_short.x + l_now.z * l_short.z, -1.0, 1.0)
  local crossP = l_short.x * l_now.z - l_short.z * l_now.x
  local d_angle_deg = math.abs(math.atan2(crossP, dotP) * (180.0 / math.pi))
  yaw_rate = d_angle_deg / dt_short

  local yaw_accel = 0
  if n >= 3 then
    local s_prior = state.samples[n - 2]
    local dt_p = s_short.t - s_prior.t
    if dt_p <= 0.0001 then dt_p = 0.03 end
    local dotP_p = clamp(l_short.x * s_prior.look.x + l_short.z * s_prior.look.z, -1.0, 1.0)
    local crossP_p = s_prior.look.x * l_short.z - s_prior.look.z * l_short.x
    local prior_yaw_rate = math.abs(math.atan2(crossP_p, dotP_p) * (180.0 / math.pi)) / dt_p
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
-- Event Importance & Detection
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

local function evaluateEventRules(f, curEv)
  local prevType = curEv and curEv.event_type or 'CRUISE'
  local inCorneringSequence = (state.macro_state == 'CORNER_SEQUENCE') or
                             (prevType == 'TURN_IN' or prevType == 'CORNER' or prevType == 'APEX')

  local proposals = {}

  if f.accel < -2.0 then
    local conf = math.min(0.98, 0.60 + (math.abs(f.accel) - 2.0) * 0.09)
    if f.accel < -4.2 then conf = math.max(0.88, conf) end
    proposals['BRAKING'] = conf
  end

  if f.accel > 1.0 and f.yaw_rate < 15.0 then
    local conf = math.min(0.96, 0.62 + (f.accel - 1.0) * 0.12)
    if prevType == 'EXIT_CORNER' then conf = conf + 0.05 end
    proposals['ACCELERATION'] = math.min(0.98, conf)
  end

  if f.yaw_rate > 11.0 and f.yaw_accel > 5.0 and f.accel < 1.2 then
    local conf = math.min(0.93, 0.65 + (f.yaw_rate - 11.0) * 0.015)
    if prevType == 'BRAKING' or prevType == 'APPROACH_CORNER' or prevType == 'CRUISE' then
      conf = conf + 0.06
    end
    proposals['TURN_IN'] = math.min(0.95, conf)
  end

  if inCorneringSequence and (not state.apex_latched) and (curEv.duration or 0) >= 0.25 then
    local isYawPeak = (f.yaw_rate > 14.0 and f.yaw_accel <= 3.0)
    local isSpeedMin = (f.accel >= -1.0 and f.accel <= 0.8 and f.yaw_rate > 13.0)
    if (isYawPeak or isSpeedMin) and prevType ~= 'APEX' and prevType ~= 'EXIT_CORNER' then
      local conf = 0.84
      if isYawPeak and isSpeedMin then conf = 0.92 end
      proposals['APEX'] = conf
    end
  end

  if (f.yaw_rate > 13.0 or (f.curv_current > 0.035 and f.yaw_rate > 8.5)) and f.accel < 1.6 then
    local conf = math.min(0.95, 0.68 + (f.yaw_rate - 13.0) * 0.012)
    if inCorneringSequence then conf = conf + 0.05 end
    proposals['CORNER'] = math.min(0.96, conf)
  end

  if (prevType == 'APEX' or prevType == 'CORNER') and f.yaw_rate < 12.0 and f.yaw_accel < 0.0 and f.accel > -1.0 then
    local conf = math.min(0.88, 0.72 + (12.0 - f.yaw_rate) * 0.015)
    proposals['EXIT_CORNER'] = conf
  end

  if f.curv_ahead > 0.045 and f.curv_current < 0.025 and f.yaw_rate < 7.5 and f.speed_kmh > 45.0 and f.accel > -2.0 then
    local conf = math.min(0.86, 0.65 + (f.curv_ahead - 0.045) * 3.5)
    proposals['APPROACH_CORNER'] = conf
  end

  if f.yaw_rate < 7.0 and math.abs(f.accel) < 1.1 and f.curv_ahead < 0.035 then
    local conf = math.min(0.95, 0.82 + (7.0 - f.yaw_rate) * 0.018)
    proposals['CRUISE'] = conf
  end

  local bestType = 'CRUISE'
  local bestConf = 0.50

  for etype, conf in pairs(proposals) do
    if etype == prevType then
      conf = conf + 0.06
    end
    if conf > bestConf then
      bestConf = conf
      bestType = etype
    end
  end

  return bestType, math.min(1.0, math.floor(bestConf * 100 + 0.5) / 100)
end

local function updateDrivingPhase(kinematics)
  local curEv = state.driving_event
  local rawType, rawConf = evaluateEventRules(kinematics, curEv)

  local curDuration = (curEv.start_time > 0) and (kinematics.t - curEv.start_time) or 0
  curEv.duration = curDuration

  local minFrames = DEFAULT_MIN_CONFIRM_FRAMES
  if rawType == 'BRAKING' and kinematics.accel < -3.8 then
    minFrames = FAST_CONFIRM_FRAMES
  end

  local selectedType = curEv.event_type
  local selectedConf = curEv.confidence

  if rawType ~= curEv.event_type then
    if state.pending_state == rawType then
      state.pending_frames = state.pending_frames + 1
      if state.pending_frames >= minFrames then
        local minDur = (curEv.event_type == 'APEX') and APEX_MIN_DURATION_S or DEFAULT_MIN_DURATION_S
        local isUrgentBraking = (rawType == 'BRAKING' and kinematics.accel < -3.5)

        if curDuration >= minDur or isUrgentBraking then
          selectedType = rawType
          selectedConf = rawConf
          state.pending_state = nil
          state.pending_frames = 0
        end
      end
    else
      state.pending_state = rawType
      state.pending_frames = 1
    end
  else
    state.pending_state = nil
    state.pending_frames = 0
    selectedConf = rawConf
  end

  if selectedType ~= curEv.event_type then
    curEv.event_type = selectedType
    curEv.confidence = selectedConf
    curEv.start_time = kinematics.t
    curEv.start_frame = kinematics.frame
    curEv.duration = 0
  else
    curEv.confidence = selectedConf
  end

  curEv.importance = calculateEventImportance(curEv.event_type, kinematics)

  if curEv.event_type == 'TURN_IN' or curEv.event_type == 'CORNER' or curEv.event_type == 'APEX' then
    state.macro_state = 'CORNER_SEQUENCE'
    if curEv.event_type == 'APEX' then
      state.apex_latched = true
    end
  elseif curEv.event_type == 'EXIT_CORNER' or curEv.event_type == 'ACCELERATION' or curEv.event_type == 'CRUISE' then
    if kinematics.yaw_rate < 8.0 and (curEv.duration or 0) > 0.40 then
      state.macro_state = 'STRAIGHT_RUN'
      state.apex_latched = false
    end
  end
end

-- ============================================================================
-- Motion Profile Configuration & Generator
-- ============================================================================

local function configureMotionProfile(profileType, carPos, carLook, carSide, carSpeed, eventType, importance, candPos)
  local motion = {
    type = profileType,
    start_time = state.samples[#state.samples].t,
    duration = 3.0,
    intensity = 1.0,
    easing = 'easeInOutQuad',
    fov_start = 52.0,
    fov_end = 46.0,
    dutch_max_deg = 2.2,
    mode = 'world', -- 'world' or 'car_relative'

    -- Position anchors
    anchor_pos = candPos or vec3(carPos.x, carPos.y + 1.2, carPos.z),
    push_vector = vec3(0, 0, 1),
    travel_dist = 12.0,

    -- Relative offsets (car frame)
    rel_lateral = 3.6,
    rel_height = 0.95,
    rel_long_start = 1.0,
    rel_long_end = -1.2,
    chase_dist = 6.8,
    chase_offset_start = -1.0,
    chase_offset_end = 1.0,
    orbit_radius = 7.5,
    orbit_angle_start = 30.0,
    orbit_angle_end = 110.0,

    -- Live smoothing state
    smooth_pos = nil,
    smooth_look = nil,
  }

  local sideSign = (math.random() > 0.5) and 1.0 or -1.0
  if state.live.lateral_m > 0.5 then sideSign = -1.0 end
  if state.live.lateral_m < -0.5 then sideSign = 1.0 end

  if profileType == 'DOLLY_IN' then
    motion.mode = 'world'
    motion.duration = (eventType == 'APEX') and 1.5 or 2.5
    motion.fov_start = 54.0
    motion.fov_end = 43.0
    motion.easing = 'easeOutQuad'
    local toCar = v3norm(vec3(carPos.x - motion.anchor_pos.x, 0, carPos.z - motion.anchor_pos.z))
    motion.push_vector = toCar
    local initDist = v3dist(motion.anchor_pos, carPos)
    motion.travel_dist = math.min(14.0, math.max(4.0, initDist * 0.42))

  elseif profileType == 'DOLLY_OUT' then
    motion.mode = 'world'
    motion.duration = 3.2
    motion.fov_start = 45.0
    motion.fov_end = 56.0
    motion.easing = 'easeOutQuad'
    local fromCar = v3norm(vec3(motion.anchor_pos.x - carPos.x, 0, motion.anchor_pos.z - carPos.z))
    motion.push_vector = fromCar
    motion.travel_dist = 11.0

  elseif profileType == 'SIDE_TRACK' then
    motion.mode = 'car_relative'
    motion.duration = 3.4
    motion.fov_start = 48.0
    motion.fov_end = 52.0
    motion.easing = 'easeInOutQuad'
    motion.rel_lateral = sideSign * 3.6
    motion.rel_height = 0.90 -- low angle flank
    motion.rel_long_start = 0.8
    motion.rel_long_end = -1.4

  elseif profileType == 'CHASE' then
    motion.mode = 'car_relative'
    motion.duration = 3.8
    motion.fov_start = 52.0
    motion.fov_end = 56.0
    motion.easing = 'easeInOutCubic'
    motion.rel_height = 1.25
    motion.chase_dist = 6.8
    motion.chase_offset_start = -1.2 * sideSign
    motion.chase_offset_end = 1.2 * sideSign

  elseif profileType == 'ORBIT' then
    motion.mode = 'car_relative'
    motion.duration = 3.2
    motion.fov_start = 49.0
    motion.fov_end = 45.0
    motion.easing = 'easeInOutQuad'
    motion.orbit_radius = 7.2
    motion.rel_height = 1.20
    if sideSign > 0 then
      motion.orbit_angle_start = 25.0
      motion.orbit_angle_end = 115.0
    else
      motion.orbit_angle_start = -25.0
      motion.orbit_angle_end = -115.0
    end

  elseif profileType == 'FLY_BY' then
    motion.mode = 'world'
    motion.duration = 3.6
    motion.fov_start = 54.0
    motion.fov_end = 46.0
    motion.easing = 'easeInOutQuad'
    local toCar = v3norm(vec3(carPos.x - motion.anchor_pos.x, 0, carPos.z - motion.anchor_pos.z))
    motion.push_vector = toCar
    motion.travel_dist = 3.0 -- subtle drift
  end

  return motion
end

-- ============================================================================
-- Camera Candidate Placement & Scoring
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

local function selectMotionProfileForEvent(eventType, importance, category, angleType)
  if eventType == 'BRAKING' then
    return 'DOLLY_IN'
  elseif eventType == 'APEX' then
    return (category == 'low') and 'DOLLY_IN' or 'ORBIT'
  elseif eventType == 'TURN_IN' then
    return 'ORBIT'
  elseif eventType == 'CORNER' then
    return 'SIDE_TRACK'
  elseif eventType == 'EXIT_CORNER' then
    return 'DOLLY_OUT'
  elseif eventType == 'ACCELERATION' then
    if angleType == 'receding' then
      return 'CHASE'
    else
      return 'FLY_BY'
    end
  elseif eventType == 'APPROACH_CORNER' then
    return 'DOLLY_IN'
  else -- CRUISE
    if math.random() > 0.5 then
      return 'CHASE'
    else
      return 'FLY_BY'
    end
  end
end

local function scoreCandidate(c, currentShot, carPos, carLook, eventType, importance)
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

  local score = 25.0
  local d = c.distance_to_car
  if d >= 10.0 and d <= 30.0 then
    score = score + 18.0
  elseif d >= 5.0 and d < 10.0 then
    score = score + 14.0
  else
    score = score + 8.0
  end

  -- Event matching
  if eventType == 'BRAKING' and angleType == 'oncoming' then score = score + 22.0 end
  if eventType == 'CORNER' and (c.category == 'low' or c.category == 'roadside') then score = score + 20.0 end
  if eventType == 'APEX' and c.category == 'low' then score = score + 24.0 end
  if eventType == 'ACCELERATION' and angleType == 'receding' then score = score + 20.0 end

  -- Angular diversity
  if currentShot and currentShot.camera_position then
    local curPos = vec3(currentShot.camera_position[1], currentShot.camera_position[2], currentShot.camera_position[3])
    local curToCam = v3norm(vec3(curPos.x - carPos.x, 0, curPos.z - carPos.z))
    local dotP = clamp(v3dot(curToCam, toCam), -1.0, 1.0)
    local diffDeg = math.acos(dotP) * (180.0 / math.pi)
    if diffDeg < 40.0 then
      score = score - 24.0
    elseif diffDeg >= 75.0 then
      score = score + 12.0
    end
  end

  -- Recent history penalty
  if state.recent_cameras then
    for rIdx, recentType in ipairs(state.recent_cameras) do
      if c.type == recentType then
        score = score - ((rIdx == 1) and 22.0 or 12.0)
      end
    end
  end

  c.calculated_score = math.max(10.0, math.min(100.0, score))
  c.angle_type = angleType
  return c.calculated_score
end

-- ============================================================================
-- Shot Execution & Transition
-- ============================================================================

local function executeCut(bestCand, cutReason, currentFrame, replayTime, carPos, drivingEvent, forcedProfile)
  local car = ac.getCar(0)
  local carLook = car.look and vec3(car.look.x, car.look.y, car.look.z) or vec3(0, 0, 1)
  local carSide = car.side and vec3(car.side.x, car.side.y, car.side.z) or vec3(1, 0, 0)
  local carSpeed = car.speedKmh or 50.0

  local candPos = bestCand and bestCand.final_pos or vec3(carPos.x - carLook.x * 6.0, carPos.y + 1.2, carPos.z - carLook.z * 6.0)
  local chosenProfile = forcedProfile or selectMotionProfileForEvent(
    drivingEvent.event_type,
    drivingEvent.importance,
    bestCand and bestCand.category or 'roadside',
    bestCand and bestCand.angle_type or 'side_pan'
  )

  local motion = configureMotionProfile(
    chosenProfile,
    carPos,
    carLook,
    carSide,
    carSpeed,
    drivingEvent.event_type,
    drivingEvent.importance,
    candPos
  )

  -- Apply manual overrides if in manual test mode
  if state.ui_mode == 2 then
    motion.type = state.manual.profile_type
    motion.intensity = state.manual.intensity
    motion.duration = state.manual.duration
    motion.fov_start = state.manual.fov_start
    motion.fov_end = state.manual.fov_end
    motion.easing = state.manual.easing
    if state.manual.side == 'right' then
      motion.rel_lateral = math.abs(motion.rel_lateral)
    else
      motion.rel_lateral = -math.abs(motion.rel_lateral)
    end
  end

  local newShot = {
    shot_index = #state.session_shots + 1,
    start_frame = currentFrame,
    start_time = replayTime,
    target_duration = motion.duration,
    cut_reason = cutReason,
    event_type = drivingEvent.event_type,
    event_importance = drivingEvent.importance,
    macro_state = state.macro_state,
    camera_type = bestCand and bestCand.type or 'dynamic_relative',
    category = bestCand and bestCand.category or 'relative',
    angle_type = bestCand and bestCand.angle_type or 'side_pan',
    camera_position = { candPos.x, candPos.y, candPos.z },
    motion_profile = motion,
  }

  state.current_shot = newShot
  state.shot_start_frame = currentFrame
  state.shot_start_time = replayTime
  state.last_cut_reason = cutReason

  if bestCand then
    table.insert(state.recent_cameras, 1, bestCand.type)
    while #state.recent_cameras > 3 do
      table.remove(state.recent_cameras)
    end
  end

  table.insert(state.session_shots, newShot)
  ac.log(string.format('[DynamicCamera] CUT #%d at frame %d (t=%.2fs) -> %s [%s] reason=%s',
    newShot.shot_index, currentFrame, replayTime, newShot.camera_type, motion.type, cutReason))
end

-- ============================================================================
-- Pacing & Cut Decision Logic
-- ============================================================================

local function evaluateCutDecision(currentFrame, replayTime, currentShot, drivingEvent, candidates)
  if currentShot == nil then
    return true, 'INIT'
  end

  local timeInShot = replayTime - currentShot.start_time
  local motion = currentShot.motion_profile

  -- If shot reached its planned motion duration
  if timeInShot >= motion.duration then
    return true, 'DURATION_EXPIRED'
  end

  local isStraight = (state.macro_state == 'STRAIGHT_RUN')
  local cooldownSec = isStraight and 2.0 or 1.5

  -- Early exit check
  if timeInShot < cooldownSec then
    return false, nil
  end

  -- Event change cut
  if drivingEvent.event_type ~= currentShot.event_type then
    local allowCut = false
    if isStraight then
      if timeInShot >= 2.4 and drivingEvent.importance >= 0.60 then
        allowCut = true
      end
    else
      -- In corner sequence: ignore throttle tap fluctuations
      local isPedalTap = (currentShot.event_type == 'CORNER' and (drivingEvent.event_type == 'ACCELERATION' or drivingEvent.event_type == 'BRAKING'))
      if not isPedalTap then
        local minDur = (currentShot.event_type == 'APEX') and MIN_SHOT_DURATION_APEX or 2.8
        if timeInShot >= minDur and drivingEvent.importance >= 0.65 then
          allowCut = true
        end
      end
    end

    if allowCut then
      return true, 'EVENT_TRANSITION_' .. drivingEvent.event_type
    end
  end

  return false, nil
end

-- ============================================================================
-- Per-Frame Dynamic Camera Motion Evaluator
-- ============================================================================

local function updateDynamicCameraTransform(carPos, carLook, carSide, carSpeed, dt, replayTime)
  if state.director_cam == nil or state.current_shot == nil then return end

  local shot = state.current_shot
  local motion = shot.motion_profile
  if motion == nil then return end

  local timeInShot = replayTime - shot.start_time
  local tNorm = clamp(timeInShot / motion.duration, 0.0, 1.0)
  local p = applyEasing(tNorm, motion.easing)

  local rawCamPos = nil
  local targetLookPos = vec3(carPos.x, carPos.y + 0.60, carPos.z)
  local liveFov = motion.fov_start + (motion.fov_end - motion.fov_start) * p

  -- 1. Compute Raw Camera Position based on Profile
  if motion.type == 'DOLLY_IN' then
    local pushDist = motion.travel_dist * p * motion.intensity
    rawCamPos = vec3(
      motion.anchor_pos.x + motion.push_vector.x * pushDist,
      motion.anchor_pos.y + motion.push_vector.y * pushDist,
      motion.anchor_pos.z + motion.push_vector.z * pushDist
    )

  elseif motion.type == 'DOLLY_OUT' then
    local pullDist = motion.travel_dist * p * motion.intensity
    rawCamPos = vec3(
      motion.anchor_pos.x + motion.push_vector.x * pullDist,
      motion.anchor_pos.y + (pullDist * 0.20),
      motion.anchor_pos.z + motion.push_vector.z * pullDist
    )

  elseif motion.type == 'SIDE_TRACK' then
    local longShift = (motion.rel_long_start + (motion.rel_long_end - motion.rel_long_start) * p) * motion.intensity
    local targetRel = vec3(
      carPos.x + carSide.x * motion.rel_lateral + carLook.x * longShift,
      carPos.y + motion.rel_height,
      carPos.z + carSide.z * motion.rel_lateral + carLook.z * longShift
    )
    if motion.smooth_pos == nil then
      motion.smooth_pos = targetRel
    else
      local alpha = clamp(14.0 * dt, 0.15, 0.90)
      motion.smooth_pos = v3lerp(motion.smooth_pos, targetRel, alpha)
    end
    rawCamPos = motion.smooth_pos
    targetLookPos = vec3(carPos.x, carPos.y + 0.55, carPos.z)

  elseif motion.type == 'CHASE' then
    local accelMod = clamp((state.live.accel_ms2 or 0) * 0.22, -1.2, 1.8)
    local effDist = (motion.chase_dist - accelMod) * motion.intensity
    local sideShift = (motion.chase_offset_start + (motion.chase_offset_end - motion.chase_offset_start) * p) * motion.intensity
    local targetRel = vec3(
      carPos.x - carLook.x * effDist + carSide.x * sideShift,
      carPos.y + motion.rel_height,
      carPos.z - carLook.z * effDist + carSide.z * sideShift
    )
    if motion.smooth_pos == nil then
      motion.smooth_pos = targetRel
    else
      local alpha = clamp(12.0 * dt, 0.15, 0.90)
      motion.smooth_pos = v3lerp(motion.smooth_pos, targetRel, alpha)
    end
    rawCamPos = motion.smooth_pos
    targetLookPos = vec3(carPos.x + carLook.x * 0.8, carPos.y + 0.65, carPos.z + carLook.z * 0.8)

  elseif motion.type == 'ORBIT' then
    local currentDeg = motion.orbit_angle_start + (motion.orbit_angle_end - motion.orbit_angle_start) * p
    local rad = currentDeg * (math.pi / 180.0)
    local arcDist = motion.orbit_radius * motion.intensity
    local arcOffset = vec3(
      carLook.x * math.cos(rad) + carSide.x * math.sin(rad),
      0,
      carLook.z * math.cos(rad) + carSide.z * math.sin(rad)
    )
    local targetRel = vec3(
      carPos.x + arcOffset.x * arcDist,
      carPos.y + motion.rel_height,
      carPos.z + arcOffset.z * arcDist
    )
    if motion.smooth_pos == nil then
      motion.smooth_pos = targetRel
    else
      local alpha = clamp(12.0 * dt, 0.15, 0.90)
      motion.smooth_pos = v3lerp(motion.smooth_pos, targetRel, alpha)
    end
    rawCamPos = motion.smooth_pos
    targetLookPos = vec3(carPos.x, carPos.y + 0.60, carPos.z)

  elseif motion.type == 'FLY_BY' then
    local driftDist = motion.travel_dist * p * motion.intensity
    rawCamPos = vec3(
      motion.anchor_pos.x - carLook.x * driftDist,
      motion.anchor_pos.y,
      motion.anchor_pos.z - carLook.z * driftDist
    )
    local leadTime = 0.04
    targetLookPos = vec3(
      carPos.x + carLook.x * (carSpeed / 3.6) * leadTime,
      carPos.y + 0.60,
      carPos.z + carLook.z * (carSpeed / 3.6) * leadTime
    )
  end

  if rawCamPos == nil then rawCamPos = motion.anchor_pos end

  -- 2. Obstacle & Ground Collision Clamping
  local groundY = findGround(rawCamPos)
  if groundY ~= nil then
    if rawCamPos.y < groundY + MIN_CLEARANCE then
      rawCamPos = vec3(rawCamPos.x, groundY + MIN_CLEARANCE, rawCamPos.z)
    end
  end

  -- Distance safety: ensure >= 2.2m from car
  local distToCar = v3dist(rawCamPos, carPos)
  if distToCar < 2.2 then
    local pushDir = v3norm(vec3(rawCamPos.x - carPos.x, 0.2, rawCamPos.z - carPos.z))
    rawCamPos = vec3(carPos.x + pushDir.x * 2.2, rawCamPos.y, carPos.z + pushDir.z * 2.2)
    distToCar = 2.2
  end

  -- 3. Compute Orthonormal Orientation with Subtle Dutch Roll
  local look = v3norm(vec3(targetLookPos.x - rawCamPos.x, targetLookPos.y - rawCamPos.y, targetLookPos.z - rawCamPos.z))
  local worldUp = vec3(0, 1, 0)
  if math.abs(v3dot(look, worldUp)) > 0.999 then worldUp = vec3(0, 0, 1) end

  local right = v3norm(v3cross(look, worldUp))
  local up = v3cross(right, look)

  -- Subtle Dutch Roll Banking during Cornering
  local rollDeg = 0.0
  if state.live.yaw_rate_dps and math.abs(state.live.yaw_rate_dps) > 5.0 then
    local rollSign = (state.live.lateral_m > 0) and -1.0 or 1.0
    rollDeg = clamp(rollSign * (state.live.yaw_rate_dps / 30.0) * motion.dutch_max_deg, -motion.dutch_max_deg, motion.dutch_max_deg)
    local rollRad = rollDeg * (math.pi / 180.0)
    local cosR, sinR = math.cos(rollRad), math.sin(rollRad)
    local newRight = vec3(right.x * cosR + up.x * sinR, right.y * cosR + up.y * sinR, right.z * cosR + up.z * sinR)
    local newUp = vec3(-right.x * sinR + up.x * cosR, -right.y * sinR + up.y * cosR, -right.z * sinR + up.z * cosR)
    right = newRight
    up = newUp
  end

  -- 4. Apply to CSP Camera
  state.director_cam.transform.position = rawCamPos
  state.director_cam.transform.look = look
  state.director_cam.transform.side = right
  state.director_cam.transform.up = up
  state.director_cam.fov = liveFov

  -- 5. Record HUD telemetry
  state.live_motion.profile_name = motion.type
  state.live_motion.progress_norm = tNorm
  state.live_motion.eased_progress = p
  state.live_motion.current_fov = liveFov
  state.live_motion.cam_dist = distToCar
  state.live_motion.cam_pos = rawCamPos
  state.live_motion.target_pos = targetLookPos
  state.live_motion.roll_deg = rollDeg
end

-- ============================================================================
-- Main Replay Processing Loop
-- ============================================================================

local function processDirectorFrame(dt)
  if not state.director_active then return end

  local sim = ac.getSim()
  if sim == nil then return end

  local currentFrame = sim.replayCurrentFrame or 0
  local isPaused = (sim.replayPlaybackRate == 0)

  local car = ac.getCar(0)
  if car == nil or car.position == nil then return end

  local carPos = vec3(car.position.x, car.position.y, car.position.z)
  local carLook = car.look and vec3(car.look.x, car.look.y, car.look.z) or vec3(0, 0, 1)
  local carSide = car.side and vec3(car.side.x, car.side.y, car.side.z) or vec3(1, 0, 0)
  local carSpeed = car.speedKmh or 50.0
  local carProgress = car.splinePosition or 0
  local replayTime = currentFrame / state.sample_rate

  -- Scrub / Jump Detection
  if state.last_processed_frame ~= nil then
    if math.abs(currentFrame - state.last_processed_frame) > 40 then
      ac.log(string.format('[DynamicCamera] Scrub detected (%d -> %d). Re-syncing.',
        state.last_processed_frame, currentFrame))
      state.samples = {}
      state.current_shot = nil
    end
  end

  if isPaused or currentFrame == state.last_processed_frame then
    updateDynamicCameraTransform(carPos, carLook, carSide, carSpeed, dt, replayTime)
    return
  end

  state.last_processed_frame = currentFrame

  -- Ingest sample
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

  -- Compute kinematics
  do
    local k = computeKinematics()
    if k ~= nil then
      updateDrivingPhase(k)
    end
  end

  -- In Autonomous Director Mode, evaluate candidates and cut decisions
  if state.ui_mode == 1 then
    local candidates = generateCandidates(carPos, carProgress)
    if #candidates > 0 then
      for _, cand in ipairs(candidates) do
        scoreCandidate(cand, state.current_shot, carPos, carLook, state.driving_event.event_type, state.driving_event.importance)
      end
      table.sort(candidates, function(a, b) return (a.calculated_score or 0) > (b.calculated_score or 0) end)
    end

    local shouldCut, cutReason = evaluateCutDecision(currentFrame, replayTime, state.current_shot, state.driving_event, candidates)
    if shouldCut then
      local best = (#candidates > 0) and candidates[1] or nil
      executeCut(best, cutReason, currentFrame, replayTime, carPos, state.driving_event)
    end
  elseif state.ui_mode == 2 then
    -- In Manual Mode, if no shot is active yet, initialize one
    if state.current_shot == nil then
      executeCut(nil, 'MANUAL_INIT', currentFrame, replayTime, carPos, state.driving_event, state.manual.profile_type)
    end
  end

  -- Apply dynamic camera motion
  updateDynamicCameraTransform(carPos, carLook, carSide, carSpeed, dt, replayTime)
end

-- ============================================================================
-- Start / Stop Director
-- ============================================================================

local function startDirector()
  if state.director_active then return end

  local ok, cam = pcall(function()
    return ac.grabCamera('Dynamic Camera Director')
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
    start_frame = state.session_start_frame or 0,
    duration = 0,
  }
  state.status_message = 'Dynamic Director ACTIVE (Procedural Motion Engaged)'
  ac.log('[DynamicCamera] Director started at frame ' .. tostring(state.session_start_frame))
end

local function stopDirector()
  if not state.director_active then return end

  if state.director_cam then
    pcall(function() state.director_cam:dispose() end)
    state.director_cam = nil
  end

  state.director_active = false
  state.current_shot = nil
  state.status_message = 'Dynamic Director STOPPED'
  ac.log(string.format('[DynamicCamera] Director stopped. Total dynamic shots: %d', #state.session_shots))
end

-- ============================================================================
-- JSON Diagnostics Exporter
-- ============================================================================

local function exportMotionReport()
  local report = {
    version = '1.0.0',
    timestamp = os.time(),
    csp_version = state.csp,
    ui_mode = (state.ui_mode == 1) and 'AUTONOMOUS' or 'MANUAL_TEST',
    total_shots = #state.session_shots,
    evaluation_stats = {
      candidates_evaluated = state.eval_count,
      candidates_accepted = state.eval_accepted,
      candidates_rejected = state.eval_rejected,
    },
    shots = {},
  }

  for i, s in ipairs(state.session_shots) do
    local m = s.motion_profile or {}
    report.shots[i] = {
      shot_index = s.shot_index,
      start_frame = s.start_frame,
      start_time = s.start_time,
      target_duration = s.target_duration,
      cut_reason = s.cut_reason,
      event_type = s.event_type,
      event_importance = s.event_importance,
      macro_state = s.macro_state,
      camera_type = s.camera_type,
      category = s.category,
      angle_type = s.angle_type,
      camera_position = s.camera_position,
      motion = {
        type = m.type,
        mode = m.mode,
        duration = m.duration,
        intensity = m.intensity,
        easing = m.easing,
        fov_start = m.fov_start,
        fov_end = m.fov_end,
      }
    }
  end

  local reportJson = json.encode(report)
  local path = 'dynamic_camera_report.json'
  local f, err = io.open(path, 'w')
  if f then
    f:write(reportJson)
    f:close()
    state.save_message = string.format('Saved report: %d shots to %s', #state.session_shots, path)
  else
    state.save_message = 'Failed to write report: ' .. tostring(err)
  end
end

-- ============================================================================
-- Initialization
-- ============================================================================

local function probeEnvironment()
  if state.probed then return end
  state.probed = true

  if ac.getPatchVersionCode then
    state.csp = ac.getPatchVersionCode()
  else
    state.csp = 'unknown'
  end

  local trackLen = ac.getTrackLength and ac.getTrackLength() or 0
  if trackLen > 100 then
    state.track_length_m = trackLen
  end
end

-- ============================================================================
-- UI Window Main
-- ============================================================================

function windowMain(dt)
  probeEnvironment()
  processDirectorFrame(dt)

  local sim = ac.getSim()
  local replayFrame = sim and sim.replayCurrentFrame or 0
  local replayTime = replayFrame / state.sample_rate

  ui.pushFont(ui.Font.Title)
  ui.text('Dynamic Camera Spike')
  ui.popFont()
  ui.sameLine()
  ui.textColored(string.format('(CSP: %s)', tostring(state.csp)), rgbm(0.6, 0.6, 0.6, 1.0))

  ui.separator()

  -- Mode Selector Tabs
  ui.pushStyleVar(ui.StyleVar.FramePadding, vec2(10, 5))
  if ui.radioButton('Autonomous Dynamic Director', state.ui_mode == 1) then
    state.ui_mode = 1
  end
  ui.sameLine()
  if ui.radioButton('Manual Profile Tester', state.ui_mode == 2) then
    state.ui_mode = 2
  end
  ui.popStyleVar()

  ui.separator()

  -- Control Buttons
  if not state.director_active then
    if ui.button('START DYNAMIC DIRECTOR', vec2(220, 36)) then
      startDirector()
    end
  else
    if ui.button('STOP DYNAMIC DIRECTOR', vec2(220, 36)) then
      stopDirector()
    end
  end

  ui.sameLine()
  if ui.button('Save Motion Report', vec2(160, 36)) then
    exportMotionReport()
  end

  ui.sameLine()
  if ui.button('Reset History', vec2(120, 36)) then
    state.session_shots = {}
    state.current_shot = nil
    state.save_message = 'History cleared.'
  end

  if state.status_message then
    local statusColor = state.director_active and rgbm(0.2, 0.9, 0.3, 1) or rgbm(0.9, 0.5, 0.2, 1)
    ui.textColored(state.status_message, statusColor)
  end
  if state.save_message then
    ui.textColored(state.save_message, rgbm(0.4, 0.8, 1, 1))
  end

  ui.separator()

  -- ==========================================
  -- Tab 1: Autonomous Dynamic Director
  -- ==========================================
  if state.ui_mode == 1 then
    -- Live Driving Event & Macro State
    local ev = state.driving_event
    local evColor = rgbm(0.3, 0.8, 1.0, 1.0)
    if ev.event_type == 'BRAKING' then evColor = rgbm(1.0, 0.3, 0.3, 1.0)
    elseif ev.event_type == 'APEX' then evColor = rgbm(1.0, 0.9, 0.1, 1.0)
    elseif ev.event_type == 'TURN_IN' then evColor = rgbm(1.0, 0.6, 0.2, 1.0)
    elseif ev.event_type == 'CORNER' then evColor = rgbm(0.8, 0.4, 1.0, 1.0)
    elseif ev.event_type == 'EXIT_CORNER' then evColor = rgbm(0.3, 1.0, 0.6, 1.0)
    elseif ev.event_type == 'ACCELERATION' then evColor = rgbm(0.2, 1.0, 0.3, 1.0)
    end

    ui.text('Event: ')
    ui.sameLine()
    ui.textColored(string.format('%s (conf: %.0f%%, imp: %.2f)', ev.event_type, ev.confidence * 100, ev.importance), evColor)

    ui.sameLine(340)
    local macroColor = (state.macro_state == 'CORNER_SEQUENCE') and rgbm(1.0, 0.7, 0.2, 1) or rgbm(0.5, 0.8, 1, 1)
    ui.textColored(string.format('[%s%s]', state.macro_state, state.apex_latched and ' (APEX LATCHED)' or ''), macroColor)

    -- Live Telemetry Bar
    ui.text(string.format('Speed: %.1f km/h | Accel: %+.2f m/s² (%+.2f G) | Yaw: %.1f °/s | Curv: %.4f',
      state.live.speed_kmh, state.live.accel_ms2, state.live.accel_g, state.live.yaw_rate_dps, state.live.curv_current))

    ui.separator()

    -- Active Motion Profile HUD
    if state.current_shot and state.director_active then
      local m = state.current_shot.motion_profile or {}
      ui.textColored(string.format('ACTIVE MOTION: %s  (Mode: %s)', tostring(m.type), tostring(m.mode)), rgbm(0.2, 1.0, 0.8, 1))

      local progressPct = (state.live_motion.progress_norm or 0) * 100
      local timeInShot = replayTime - state.current_shot.start_time
      ui.progressBar(state.live_motion.progress_norm or 0, vec2(580, 18),
        string.format('Shot #%d: %.1fs / %.1fs (%.0f%%)', state.current_shot.shot_index, timeInShot, m.duration or 3.0, progressPct))

      ui.text(string.format('Live FOV: %.1f° (%s -> %s) | Cam Dist: %.1f m | Roll: %+.1f° | Easing: %s',
        state.live_motion.current_fov or 50.0,
        tostring(m.fov_start or 50),
        tostring(m.fov_end or 50),
        state.live_motion.cam_dist or 0.0,
        state.live_motion.roll_deg or 0.0,
        tostring(m.easing or 'linear')))
    else
      ui.textColored('No active dynamic shot (Director idle)', rgbm(0.6, 0.6, 0.6, 1))
    end

  -- ==========================================
  -- Tab 2: Manual Profile Tester
  -- ==========================================
  elseif state.ui_mode == 2 then
    ui.textColored('MANUAL MOTION PROFILE TESTER', rgbm(1.0, 0.8, 0.2, 1))
    ui.text('Select a motion profile below and click "Trigger Profile Now" to test immediately:')

    -- Profile selector buttons
    local profiles = { 'DOLLY_IN', 'DOLLY_OUT', 'SIDE_TRACK', 'CHASE', 'ORBIT', 'FLY_BY' }
    for idx, pName in ipairs(profiles) do
      local isSel = (state.manual.profile_type == pName)
      if isSel then
        ui.pushStyleColor(ui.Col.Button, rgbm(0.2, 0.6, 0.9, 1))
      end
      if ui.button(pName, vec2(96, 28)) then
        state.manual.profile_type = pName
        -- Auto adjust standard defaults for this profile
        if pName == 'DOLLY_IN' then
          state.manual.duration = 2.5
          state.manual.fov_start = 54.0
          state.manual.fov_end = 43.0
        elseif pName == 'DOLLY_OUT' then
          state.manual.duration = 3.2
          state.manual.fov_start = 45.0
          state.manual.fov_end = 56.0
        elseif pName == 'SIDE_TRACK' then
          state.manual.duration = 3.5
          state.manual.fov_start = 48.0
          state.manual.fov_end = 52.0
        elseif pName == 'CHASE' then
          state.manual.duration = 4.0
          state.manual.fov_start = 52.0
          state.manual.fov_end = 56.0
        elseif pName == 'ORBIT' then
          state.manual.duration = 3.5
          state.manual.fov_start = 49.0
          state.manual.fov_end = 45.0
        elseif pName == 'FLY_BY' then
          state.manual.duration = 3.6
          state.manual.fov_start = 54.0
          state.manual.fov_end = 46.0
        end
      end
      if isSel then
        ui.popStyleColor()
      end
      if idx < #profiles then ui.sameLine() end
    end

    ui.dummy(vec2(0, 4))

    -- Sliders
    local intVal, intChg = ui.slider('Intensity Multiplier##mp', state.manual.intensity, 0.2, 2.5, '%.2fx')
    if intChg then state.manual.intensity = intVal end

    local durVal, durChg = ui.slider('Shot Duration (s)##mp', state.manual.duration, 1.0, 7.0, '%.1f s')
    if durChg then state.manual.duration = durVal end

    local fsVal, fsChg = ui.slider('Start FOV (deg)##mp', state.manual.fov_start, 30.0, 75.0, '%.1f°')
    if fsChg then state.manual.fov_start = fsVal end

    local feVal, feChg = ui.slider('End FOV (deg)##mp', state.manual.fov_end, 30.0, 75.0, '%.1f°')
    if feChg then state.manual.fov_end = feVal end

    -- Side selector for Side-Track & Orbit
    ui.text('Flank Side: ')
    ui.sameLine()
    if ui.radioButton('Left Flank##sd', state.manual.side == 'left') then state.manual.side = 'left' end
    ui.sameLine()
    if ui.radioButton('Right Flank##sd', state.manual.side == 'right') then state.manual.side = 'right' end

    ui.dummy(vec2(0, 6))

    -- Big Trigger Button
    if ui.button('TRIGGER PROFILE NOW', vec2(260, 36)) then
      if not state.director_active then
        startDirector()
      end
      local car = ac.getCar(0)
      if car and car.position then
        local carPos = vec3(car.position.x, car.position.y, car.position.z)
        executeCut(nil, 'MANUAL_TRIGGER', replayFrame, replayTime, carPos, state.driving_event, state.manual.profile_type)
      end
    end

    if state.current_shot and state.director_active then
      local m = state.current_shot.motion_profile or {}
      ui.dummy(vec2(0, 4))
      ui.textColored(string.format('LIVE TEST: %s (Duration: %.1fs)', tostring(m.type), m.duration or 3.0), rgbm(0.2, 1.0, 0.5, 1))
      ui.progressBar(state.live_motion.progress_norm or 0, vec2(580, 18),
        string.format('%.1fs / %.1fs (FOV: %.1f°)', replayTime - state.current_shot.start_time, m.duration or 3.0, state.live_motion.current_fov or 50.0))
    end
  end

  ui.separator()

  -- Recent Dynamic Shot History Table
  ui.text(string.format('Recorded Dynamic Shots: %d', #state.session_shots))
  ui.dummy(vec2(0, 2))

  local startIdx = math.max(1, #state.session_shots - 6)
  for i = #state.session_shots, startIdx, -1 do
    local s = state.session_shots[i]
    local m = s.motion_profile or {}
    local timeStr = string.format('%.1fs', s.start_time)
    local durStr = string.format('%.1fs', s.target_duration)
    ui.text(string.format('#%02d [%s] %-10s -> %-10s | %s (%s) | %s',
      s.shot_index,
      timeStr,
      tostring(m.type or 'STATIC'),
      tostring(s.camera_type or 'cam'),
      tostring(s.event_type or 'EV'),
      durStr,
      tostring(s.cut_reason or 'CUT')))
  end
end
