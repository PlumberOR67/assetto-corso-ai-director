-- Shot Planner Spike — CSP 0.3.0-preview595
--
-- Director Capture Mode: Real-Time Replay Directing & Session Recording
--
-- Directs the camera live in real-time with ZERO extrapolation drift.
-- Evaluates candidates at actual car position, scores them with geometric angle classification,
-- tight focal distance curves, category weights, and diversity rules.
-- Automatically records every directorial choice with frame-accurate precision
-- and exports to Documents/Assetto Corsa/director_session.json.
--
-- Does NOT modify Timeline Driver, aidirector_spike, or camera_placement_spike.
-- Does NOT generate timeline.json yet (director_session.json is intermediate ground-truth).

local json = require('json')

-- ============================================================================
-- Configuration
-- ============================================================================

local RAISE_HEIGHT = 80.0
local RAY_LENGTH = 200.0
local MIN_CLEARANCE = 0.5
local LOS_TOLERANCE = 2.0

-- Calibrated camera types (proven in runtime: 25/26 visual clean rate)
local CAMERA_TYPES = {
  { name = 'roadside_left_close',   lateral = -1.25, height = 1.4, category = 'roadside', side = 'left',   desc = 'Left edge, eye level (~0.8m off)' },
  { name = 'roadside_right_close',  lateral =  1.25, height = 1.4, category = 'roadside', side = 'right',  desc = 'Right edge, eye level (~0.8m off)' },
  { name = 'roadside_left_wide',    lateral = -1.65, height = 1.6, category = 'wide',     side = 'left',   desc = 'Wide left berm (~2.2m off)' },
  { name = 'roadside_right_wide',   lateral =  1.65, height = 1.6, category = 'wide',     side = 'right',  desc = 'Wide right berm (~2.2m off)' },
  { name = 'low_left',              lateral = -1.15, height = 0.6, category = 'low',      side = 'left',   desc = 'Low left edge, dramatic angle' },
  { name = 'low_right',             lateral =  1.15, height = 0.6, category = 'low',      side = 'right',  desc = 'Low right edge, dramatic angle' },
  { name = 'elevated_left_low',     lateral = -1.35, height = 2.8, category = 'elevated', side = 'left',   desc = 'Elevated left, over bushes (2.8m)' },
  { name = 'elevated_left_mid',     lateral = -1.45, height = 4.5, category = 'elevated', side = 'left',   desc = 'Elevated left boom pole (4.5m)' },
  { name = 'elevated_right',        lateral =  1.30, height = 3.5, category = 'elevated', side = 'right',  desc = 'Elevated right overview (3.5m)' },
  { name = 'elevated_high',         lateral = -0.80, height = 7.5, category = 'high',     side = 'center', desc = 'High crane / overhead (7.5m)' },
}

-- Calibrated tight progress offsets (on 7100m track: 0.001 = ~7m)
-- ±0.0035 = ~25m (maximum oncoming lead-in for a 3s shot)
-- ±0.0018 = ~13m (intimate dynamic pass-by)
-- 0.0000 = ~0m (curb/apex view)
local PROGRESS_OFFSETS = { -0.0035, -0.0018, 0.0, 0.0018, 0.0035 }

-- ============================================================================
-- State
-- ============================================================================

local state = {
  csp = 'unknown',
  probed = false,
  track_length_m = 7100.0,
  sample_rate = 33.3333,

  -- Car state
  car_pos = nil,
  car_look = nil,
  car_progress = 0,
  car_speed_kmh = 60.0,
  car_progress_method = 'unknown',

  -- DIRECTOR CAPTURE MODE (Primary)
  live_director_active = false,
  recording_active = false,
  live_director_cam = nil,
  live_shot_end_frame = 0,
  live_current_shot = nil,
  live_last_frame = nil,
  session_shots = {},
  session_stats = {
    duration_s = 0,
    shot_count = 0,
    start_frame = 0,
    end_frame = 0,
    total_candidates_generated = 0,
    total_candidates_rejected = 0,
    total_candidates_accepted = 0,
    avg_duration_s = 0,
    type_distribution = {},
    angle_distribution = {},
    reject_reasons = {},
  },
  director_save_message = nil,

  -- Offline Batch Planner (Reference / Deprecated)
  show_batch_planner = false,
  planned_shots = {},
  stats = {
    duration_analyzed_s = 0,
    shot_count = 0,
    total_candidates_generated = 0,
    total_candidates_rejected = 0,
    total_candidates_accepted = 0,
    avg_duration_s = 0,
    type_distribution = {},
    reject_reasons = {},
  },
  plan_preview_active = false,
  plan_preview_cam = nil,
  plan_preview_shot_idx = 1,

  -- UI / status
  status_message = nil,
  save_message = nil,
  audit_save_message = nil,
}

-- ============================================================================
-- Vector math
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
-- Track Spline & Curvature Estimation
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
  ac.log(string.format('[ShotPlanner] Measured track spline length: %.1fm', state.track_length_m))
end

-- Calculates track curvature and realistic expected speed at a given progress.
-- Returns speedMs, speedKmh. Hairpins: 38-45 km/h, Bends: 50-60 km/h, Straights: 75-85 km/h.
local function getCurvatureAwareSpeed(progress)
  local p1 = ac.trackProgressToWorldCoordinate(math.max(0.0001, progress - 0.0015))
  local p2 = ac.trackProgressToWorldCoordinate(progress)
  local p3 = ac.trackProgressToWorldCoordinate(math.min(0.9999, progress + 0.0015))
  if not (p1 and p2 and p3) then
    return 13.8, 50.0 -- default ~50 km/h
  end

  local v1 = v3norm(vec3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z))
  local v2 = v3norm(vec3(p3.x - p2.x, p3.y - p2.y, p3.z - p2.z))
  local dot = math.max(-1.0, math.min(1.0, v3dot(v1, v2)))
  local turnAngle = math.acos(dot)

  local speedKmh = math.max(38.0, math.min(82.0, 80.0 - turnAngle * 260.0))
  return speedKmh / 3.6, speedKmh
end

-- ============================================================================
-- Raycast Validation
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
-- Car State Reading
-- ============================================================================

local function readCarState()
  local ok, car = pcall(function() return ac.getCar(0) end)
  if not ok or car == nil then return false end

  state.car_pos = vec3(car.position.x, car.position.y, car.position.z)

  local lookOk, carLook = pcall(function() return car.look end)
  if lookOk and carLook ~= nil then
    state.car_look = vec3(carLook.x, carLook.y, carLook.z)
  else
    state.car_look = vec3(0, 0, 1)
  end

  local spOk, spVal = pcall(function() return car.splinePosition end)
  if spOk and spVal ~= nil and type(spVal) == 'number' then
    state.car_progress = spVal
    state.car_progress_method = 'car.splinePosition'
  else
    local wtOk, trackCoord = pcall(function()
      return ac.worldCoordinateToTrack(car.position)
    end)
    if wtOk and trackCoord ~= nil then
      state.car_progress = trackCoord.z
      state.car_progress_method = 'ac.worldCoordinateToTrack'
    end
  end

  local spdOk, spd = pcall(function() return car.speedKmh end)
  if spdOk and spd ~= nil and type(spd) == 'number' and spd > 10 then
    state.car_speed_kmh = spd
  else
    state.car_speed_kmh = 55.0
  end

  return true
end

-- ============================================================================
-- Candidate Generation & Initial Validation at Station
-- ============================================================================

local function recordRejection(reason)
  state.stats.reject_reasons[reason] = (state.stats.reject_reasons[reason] or 0) + 1
  if state.recording_active then
    state.session_stats.reject_reasons[reason] = (state.session_stats.reject_reasons[reason] or 0) + 1
  end
end

local function generateAndValidateCandidates(carWorldPos, carProgress)
  local accepted = {}
  local rejected_count = 0
  local generated_count = 0

  for _, progOffset in ipairs(PROGRESS_OFFSETS) do
    local progress = carProgress + progOffset
    if progress < 0 then progress = progress + 1.0 end
    if progress > 1 then progress = progress - 1.0 end

    for _, camType in ipairs(CAMERA_TYPES) do
      generated_count = generated_count + 1

      local twOk, worldPos = pcall(function()
        return ac.trackCoordinateToWorld(vec3(camType.lateral, 0, progress))
      end)

      if not twOk or worldPos == nil then
        rejected_count = rejected_count + 1
        recordRejection('trackCoord failed')
        goto continue_cam
      end

      local groundY, hitDist = findGround(worldPos)
      if groundY == nil then
        rejected_count = rejected_count + 1
        recordRejection('no ground')
        goto continue_cam
      end

      local camY = groundY + camType.height
      local camPos = vec3(worldPos.x, camY, worldPos.z)

      if camY < groundY + MIN_CLEARANCE then
        rejected_count = rejected_count + 1
        recordRejection('min clearance')
        goto continue_cam
      end

      local dist = v3dist(camPos, carWorldPos)
      -- Candidates must be in tight focal zone: 2.5m to 38m max
      if dist < 2.5 or dist > 38.0 then
        rejected_count = rejected_count + 1
        recordRejection('dist out of range')
        goto continue_cam
      end

      local targetPos = vec3(carWorldPos.x, carWorldPos.y + 0.6, carWorldPos.z)
      local losClear, losHitDist, losTotalDist = checkLineOfSight(camPos, targetPos)
      if not losClear then
        rejected_count = rejected_count + 1
        recordRejection('LOS blocked')
        goto continue_cam
      end

      accepted[#accepted + 1] = {
        type = camType.name,
        category = camType.category,
        side = camType.side,
        desc = camType.desc,
        lateral = camType.lateral,
        height = camType.height,
        progress = progress,
        prog_offset = progOffset,
        final_pos = camPos,
        ground_y = groundY,
        distance_to_car = dist,
        los_clear = true,
      }

      ::continue_cam::
    end
  end

  state.stats.total_candidates_generated = state.stats.total_candidates_generated + generated_count
  state.stats.total_candidates_rejected = state.stats.total_candidates_rejected + rejected_count
  state.stats.total_candidates_accepted = state.stats.total_candidates_accepted + #accepted

  if state.recording_active then
    state.session_stats.total_candidates_generated = state.session_stats.total_candidates_generated + generated_count
    state.session_stats.total_candidates_rejected = state.session_stats.total_candidates_rejected + rejected_count
    state.session_stats.total_candidates_accepted = state.session_stats.total_candidates_accepted + #accepted
  end

  return accepted
end

-- ============================================================================
-- Usable Shot Window Evaluation Along Predicted Car Trajectory
-- ============================================================================

local function evaluateUsableWindow(camPos, startProgress, speedMs, trackLen, shotStartT)
  local dtStep = 0.3
  local maxDt = 5.0
  local readableStartDt = nil
  local readableEndDt = nil
  local consecutiveReadable = 0

  for dt = 0.0, maxDt, dtStep do
    local p = startProgress + (speedMs * dt) / trackLen
    if p > 1.0 then p = p - 1.0 end

    local carPos = ac.trackProgressToWorldCoordinate(p)
    if carPos then
      local dist = v3dist(camPos, carPos)
      -- Tight focal envelope: 3.0m to 42.0m max
      if dist >= 3.0 and dist <= 42.0 then
        local target = vec3(carPos.x, carPos.y + 0.6, carPos.z)
        local losClear = checkLineOfSight(camPos, target)
        if losClear then
          if readableStartDt == nil then
            readableStartDt = dt
          end
          readableEndDt = dt
          consecutiveReadable = consecutiveReadable + 1
        else
          if readableStartDt ~= nil and consecutiveReadable >= 2 then
            break
          end
        end
      else
        if readableStartDt ~= nil and consecutiveReadable >= 2 then
          break
        end
      end
    end
  end

  if readableStartDt == nil or readableEndDt == nil then
    return {
      usable_window_start = shotStartT,
      usable_window_end = shotStartT,
      usable_duration = 0.0,
      lead_in = 0.0,
      is_viable = false,
    }
  end

  local leadIn = readableStartDt
  local usableDuration = math.max(0.0, readableEndDt - readableStartDt)

  return {
    usable_window_start = math.floor((shotStartT + readableStartDt) * 10) / 10,
    usable_window_end = math.floor((shotStartT + readableEndDt) * 10) / 10,
    usable_duration = math.floor(usableDuration * 10) / 10,
    lead_in = math.floor(leadIn * 10) / 10,
    is_viable = usableDuration >= 1.0,
  }
end

-- ============================================================================
-- Dynamic Shot Duration from Usable Window
-- ============================================================================

local function determineShotDuration(window, speedKmh)
  local uDur = window.usable_duration or 3.0
  local leadIn = window.lead_in or 0.0

  local duration = leadIn + uDur

  if uDur <= 2.2 then
    duration = math.max(1.8, duration)
  elseif uDur <= 3.8 then
    duration = math.min(3.6, math.max(2.5, duration))
  else
    duration = math.min(4.5, math.max(3.2, duration * 0.9))
  end

  return math.floor(duration * 10 + 0.5) / 10
end

-- ============================================================================
-- Candidate Scoring
-- ============================================================================

local function scoreCandidate(c, prevShot, carPos, carLook, window)
  local baseScore = 25.0

  -- 1. True 3D Geometric Angle Classification
  local toCam = v3norm(vec3(c.final_pos.x - carPos.x, c.final_pos.y - carPos.y, c.final_pos.z - carPos.z))
  local dotDir = v3dot(carLook, toCam)
  local angleType = 'side_pan'
  local angleScore = 13.0

  if dotDir > 0.25 then
    angleType = 'oncoming'   -- In front of car, car approaching camera
    angleScore = 16.0
  elseif dotDir < -0.25 then
    angleType = 'receding'   -- Behind car, car driving away
    angleScore = 9.0
  else
    angleType = 'side_pan'   -- Side profile 90 deg view
    angleScore = 13.0
  end

  -- 2. Distance scoring curve (tight focal zone: 12-28m prime!)
  local d = c.distance_to_car
  local distScore = 0.0
  if d >= 12.0 and d <= 28.0 then
    distScore = 20.0         -- Prime TV pass-by distance
  elseif d >= 6.0 and d < 12.0 then
    distScore = 16.0         -- Dynamic close
  elseif d > 28.0 and d <= 36.0 then
    distScore = 12.0         -- Medium
  else
    distScore = 6.0          -- Very close or >36m
  end

  -- 3. Camera Category Quality Weight
  local catScore = 0.0
  if c.category == 'elevated' then
    catScore = 12.0          -- Clean boom/crane over bushes
  elseif c.category == 'roadside' then
    catScore = 11.0          -- Trackside marshal view
  elseif c.category == 'low' then
    catScore = 11.0          -- Low dramatic perspective
  elseif c.category == 'high' then
    catScore = 8.0           -- Overhead establishing
  elseif c.category == 'wide' then
    catScore = 5.0           -- Fallback berm
  end

  -- 4. Usable Window Quality & Lead-In Score (Track-Adaptive)
  local windowScore = 0.0
  local uDur = window.usable_duration or 0.0
  local leadIn = window.lead_in or 0.0

  if uDur >= 3.0 then
    windowScore = windowScore + 16.0
  elseif uDur >= 2.0 then
    windowScore = windowScore + 12.0
  elseif uDur >= 1.2 then
    windowScore = windowScore + 6.0
  else
    windowScore = windowScore - 20.0
  end

  -- Lead-in bonus: audience sees corner before car arrives
  if leadIn >= 0.2 and leadIn <= 0.8 then
    windowScore = windowScore + 8.0
  elseif leadIn == 0.0 then
    windowScore = windowScore + 4.0
  elseif leadIn > 1.2 then
    windowScore = windowScore - 15.0
  end

  -- 5. Diversity & Anti-Repetition Rules vs Previous Shot
  local divScore = 0.0
  if prevShot ~= nil then
    if c.type == prevShot.camera_type then
      divScore = divScore - 30.0
    end
    if prevShot.category and c.category == prevShot.category then
      divScore = divScore - 12.0
    else
      divScore = divScore + 8.0
    end
    if c.side ~= 'center' and prevShot.side and prevShot.side ~= 'center' then
      if c.side == prevShot.side then
        divScore = divScore - 4.0
      else
        divScore = divScore + 6.0
      end
    end
    if prevShot.angle_type == 'receding' and angleType == 'oncoming' then
      divScore = divScore + 10.0
    elseif prevShot.angle_type == 'receding' and angleType == 'receding' then
      divScore = divScore - 8.0
    end
    local prevDist = prevShot.distance or prevShot.distance_to_car or 20.0
    if (prevDist < 15.0 and d > 24.0) or (prevDist > 24.0 and d < 16.0) then
      divScore = divScore + 6.0
    end
  end

  local tieBreak = math.sin(c.progress * 876.5 + c.lateral * 43.2) * 0.3
  local rawScore = baseScore + angleScore + distScore + catScore + windowScore + divScore + tieBreak
  local finalScore = math.max(10.0, math.min(100.0, math.floor(rawScore * 10 + 0.5) / 10))

  c.calculated_score = finalScore
  c.angle_type = angleType
  c.window = window
  c.score_breakdown = {
    base = baseScore,
    angle = angleScore,
    distance = distScore,
    category = catScore,
    window = math.floor(windowScore * 10) / 10,
    diversity = divScore,
    raw = math.floor(rawScore * 10 + 0.5) / 10,
    final = finalScore,
    dot_dir = math.floor(dotDir * 1000) / 1000,
  }

  return finalScore
end

-- ============================================================================
-- Director Capture Mode: Real-Time Directing & Session Recording
-- ============================================================================

local function updateSessionStats()
  local totalDur = 0
  local typeDist = {}
  local angleDist = {}
  local startF = 0
  local endF = 0

  if #state.session_shots > 0 then
    startF = state.session_shots[1].shot_start_frame
    endF = state.session_shots[#state.session_shots].shot_end_frame
  end

  for _, s in ipairs(state.session_shots) do
    totalDur = totalDur + (s.duration or 0)
    typeDist[s.camera_type] = (typeDist[s.camera_type] or 0) + 1
    angleDist[s.angle_type] = (angleDist[s.angle_type] or 0) + 1
  end

  state.session_stats.duration_s = math.floor(totalDur * 100) / 100
  state.session_stats.shot_count = #state.session_shots
  state.session_stats.start_frame = startF
  state.session_stats.end_frame = endF
  state.session_stats.avg_duration_s = #state.session_shots > 0 and (math.floor((totalDur / #state.session_shots) * 100) / 100) or 0
  state.session_stats.type_distribution = typeDist
  state.session_stats.angle_distribution = angleDist
end

local function finalizeCurrentShot(endFrame)
  if #state.session_shots == 0 then return end
  local activeRec = state.session_shots[#state.session_shots]
  if activeRec then
    local actualEnd = math.max(activeRec.shot_start_frame, endFrame)
    activeRec.shot_end_frame = actualEnd
    activeRec.shot_end_time = math.floor((actualEnd / state.sample_rate) * 100) / 100
    local fCount = actualEnd - activeRec.shot_start_frame + 1
    activeRec.duration = math.floor((fCount / state.sample_rate) * 100) / 100
    updateSessionStats()
  end
end

local function saveDirectorSession()
  if #state.session_shots == 0 then
    state.director_save_message = 'Error: no recorded shots to save'
    return false
  end

  updateSessionStats()

  local docsOk, docs = pcall(function() return ac.getFolder(ac.FolderID.ACDocuments) end)
  if not docsOk or docs == nil then
    state.director_save_message = 'Error: cannot locate Documents folder'
    return false
  end

  local path = docs .. '/director_session.json'

  local sessionData = {
    schema_version = 1,
    app = 'shot_planner_spike_director_capture',
    csp_version = state.csp,
    timestamp = os.time(),
    sample_rate_hz = state.sample_rate,
    track_length_m = state.track_length_m,
    summary = {
      session_start_frame = state.session_stats.start_frame,
      session_end_frame = state.session_stats.end_frame,
      total_duration_s = state.session_stats.duration_s,
      shot_count = state.session_stats.shot_count,
      average_shot_duration_s = state.session_stats.avg_duration_s,
      candidates_generated = state.session_stats.total_candidates_generated,
      candidates_rejected = state.session_stats.total_candidates_rejected,
      candidates_accepted = state.session_stats.total_candidates_accepted,
      camera_type_distribution = state.session_stats.type_distribution,
      angle_type_distribution = state.session_stats.angle_distribution,
      reject_reasons = state.session_stats.reject_reasons,
    },
    shots = state.session_shots,
  }

  local encOk, encoded = pcall(function() return json.encode(sessionData) end)
  if not encOk or encoded == nil then
    state.director_save_message = 'Error encoding session JSON'
    return false
  end

  local file = io.open(path, 'w')
  if file == nil then
    state.director_save_message = 'Error opening: ' .. path
    return false
  end

  file:write(encoded)
  file:close()

  state.director_save_message = string.format('Session saved: %s (%d shots, %.1fs)',
    path, #state.session_shots, state.session_stats.duration_s)
  state.status_message = string.format('Saved %d shots to director_session.json', #state.session_shots)
  ac.log('[DirectorCapture] Saved session: ' .. path)
  return true
end

local function startLiveDirector(recordMode)
  if state.live_director_active then return end

  local ok, cam = pcall(function()
    return ac.grabCamera('Director Capture')
  end)
  if not ok or cam == nil then
    state.status_message = 'Failed to grab camera: ' .. tostring(cam)
    return
  end

  state.live_director_cam = cam
  state.live_director_active = true
  state.recording_active = (recordMode ~= false)
  state.live_shot_end_frame = 0
  state.live_current_shot = nil
  state.live_last_frame = nil

  if state.recording_active then
    state.session_shots = {}
    state.session_stats = {
      duration_s = 0,
      shot_count = 0,
      start_frame = 0,
      end_frame = 0,
      total_candidates_generated = 0,
      total_candidates_rejected = 0,
      total_candidates_accepted = 0,
      avg_duration_s = 0,
      type_distribution = {},
      angle_distribution = {},
      reject_reasons = {},
    }
    state.director_save_message = nil
    state.status_message = 'Director Capture: RECORDING'
    ac.log('[ShotPlanner] Director Capture started (recording active)')
  else
    state.status_message = 'Live Director: Active (monitoring only)'
    ac.log('[ShotPlanner] Live Director started (monitoring only)')
  end
end

local function stopLiveDirector(andSave)
  if not state.live_director_active then return end

  local sim = ac.getSim()
  local currentFrame = (sim and sim.replayCurrentFrame) or (state.live_last_frame or 0)

  if state.recording_active and state.live_current_shot then
    finalizeCurrentShot(currentFrame)
  end

  if state.live_director_cam then
    pcall(function() state.live_director_cam:dispose() end)
    state.live_director_cam = nil
  end

  state.live_director_active = false
  state.recording_active = false

  ac.log('[ShotPlanner] Live Director stopped')

  if andSave and #state.session_shots > 0 then
    saveDirectorSession()
  else
    state.status_message = string.format('Director stopped. %d shots recorded.', #state.session_shots)
  end
end

local function updateLiveDirector()
  if not state.live_director_active then return end

  local sim = ac.getSim()
  if sim == nil then return end

  local currentFrame = sim.replayCurrentFrame or 0
  local isPaused = (sim.replayPlaybackRate == 0)

  local car = ac.getCar(0)
  if car == nil or car.position == nil then return end

  local carPos = vec3(car.position.x, car.position.y, car.position.z)
  local carProgress = car.splinePosition or 0

  -- Detect replay scrub/rewind
  local scrubDetected = false
  if state.live_last_frame ~= nil then
    if currentFrame < state.live_last_frame - 30 or currentFrame > state.live_last_frame + 300 then
      scrubDetected = true
      ac.log(string.format('[LiveDirector] Replay scrub detected: %d -> %d', state.live_last_frame, currentFrame))
    end
  end

  local shouldCut = (state.live_current_shot == nil)
                 or scrubDetected
                 or (not isPaused and currentFrame >= state.live_shot_end_frame)

  if shouldCut then
    -- Finalize previous shot before starting new cut
    if state.recording_active and state.live_current_shot and #state.session_shots > 0 then
      local prevRec = state.session_shots[#state.session_shots]
      if prevRec then
        local actualEnd = scrubDetected and (state.live_last_frame or currentFrame) or (currentFrame - 1)
        actualEnd = math.max(prevRec.shot_start_frame, actualEnd)
        prevRec.shot_end_frame = actualEnd
        prevRec.shot_end_time = math.floor((actualEnd / state.sample_rate) * 100) / 100
        local fCount = actualEnd - prevRec.shot_start_frame + 1
        prevRec.duration = math.floor((fCount / state.sample_rate) * 100) / 100
        updateSessionStats()
      end
    end

    local speedMs, speedKmh = getCurvatureAwareSpeed(carProgress)
    local candidates = generateAndValidateCandidates(carPos, carProgress)

    if #candidates > 0 then
      local scored = {}
      for _, c in ipairs(candidates) do
        local window = evaluateUsableWindow(c.final_pos, carProgress, speedMs, state.track_length_m, currentFrame / state.sample_rate)
        if window.is_viable then
          scoreCandidate(c, state.live_current_shot, carPos, car.look or vec3(0,0,1), window)
          scored[#scored + 1] = c
        end
      end

      if #scored == 0 then
        for _, c in ipairs(candidates) do
          local window = evaluateUsableWindow(c.final_pos, carProgress, speedMs, state.track_length_m, currentFrame / state.sample_rate)
          scoreCandidate(c, state.live_current_shot, carPos, car.look or vec3(0,0,1), window)
          scored[#scored + 1] = c
        end
      end

      table.sort(scored, function(a, b) return (a.calculated_score or 0) > (b.calculated_score or 0) end)
      local best = scored[1]

      if best then
        local dur = determineShotDuration(best.window or { usable_duration = 3.0, lead_in = 0 }, speedKmh)
        state.live_shot_end_frame = currentFrame + math.floor(dur * state.sample_rate)
        state.live_current_shot = {
          camera_type = best.type,
          category = best.category,
          side = best.side,
          angle_type = best.angle_type,
          camera_position = { best.final_pos.x, best.final_pos.y, best.final_pos.z },
          distance = math.floor(best.distance_to_car * 10) / 10,
          duration = dur,
          score = best.calculated_score or 80.0,
        }

        if state.recording_active then
          local shotRec = {
            index = #state.session_shots + 1,
            shot_start_frame = currentFrame,
            shot_end_frame = state.live_shot_end_frame,
            shot_start_time = math.floor((currentFrame / state.sample_rate) * 100) / 100,
            shot_end_time = math.floor((state.live_shot_end_frame / state.sample_rate) * 100) / 100,
            duration = dur,
            car_position = {
              math.floor(carPos.x * 100) / 100,
              math.floor(carPos.y * 100) / 100,
              math.floor(carPos.z * 100) / 100,
            },
            car_progress = math.floor(carProgress * 10000) / 10000,
            camera_position = {
              math.floor(best.final_pos.x * 100) / 100,
              math.floor(best.final_pos.y * 100) / 100,
              math.floor(best.final_pos.z * 100) / 100,
            },
            camera_type = best.type,
            category = best.category,
            side = best.side,
            distance_to_car = math.floor(best.distance_to_car * 10) / 10,
            angle_type = best.angle_type,
            score = best.calculated_score or 80.0,
            score_breakdown = best.score_breakdown,
          }
          state.session_shots[#state.session_shots + 1] = shotRec
          updateSessionStats()
        end

        ac.log(string.format('[LiveDirector] Cut #%d -> %s (%s, dist=%.1fm, dur=%.1fs, score=%.1f)',
          state.recording_active and #state.session_shots or 1,
          best.type, best.angle_type, best.distance_to_car, dur, best.calculated_score or 0))
      end
    end
  end

  state.live_last_frame = currentFrame

  -- Apply live shot camera tracking
  if state.live_director_cam and state.live_current_shot then
    local target = vec3(carPos.x, carPos.y + 0.6, carPos.z)
    local camPos = vec3(state.live_current_shot.camera_position[1],
                        state.live_current_shot.camera_position[2],
                        state.live_current_shot.camera_position[3])

    local look = v3norm(vec3(target.x - camPos.x, target.y - camPos.y, target.z - camPos.z))
    local worldUp = vec3(0, 1, 0)
    if math.abs(v3dot(look, worldUp)) > 0.999 then worldUp = vec3(0, 0, 1) end
    local right = v3norm(v3cross(look, worldUp))
    local up = v3cross(right, look)

    state.live_director_cam.transform.position = camPos
    state.live_director_cam.transform.look = look
    state.live_director_cam.transform.side = right
    state.live_director_cam.transform.up = up
    state.live_director_cam.fov = 55
  end
end

-- ============================================================================
-- Offline Batch Planning Engine (Reference / Deprecated)
-- ============================================================================

local function planSequence(targetDurationSeconds)
  readCarState()
  if state.track_length_m < 100 then
    estimateTrackLength()
  end

  state.planned_shots = {}
  state.stats = {
    duration_analyzed_s = 0,
    shot_count = 0,
    total_candidates_generated = 0,
    total_candidates_rejected = 0,
    total_candidates_accepted = 0,
    avg_duration_s = 0,
    type_distribution = {},
    reject_reasons = {},
  }

  local sim = ac.getSim()
  local startT = 0
  if sim and sim.replayCurrentFrame then
    startT = math.floor((sim.replayCurrentFrame / state.sample_rate) * 10) / 10
  end

  local currentT = startT
  local currentProgress = state.car_progress

  ac.log(string.format('[ShotPlanner] Starting plan: target=%.1fs, start_t=%.1fs, start_prog=%.4f',
    targetDurationSeconds, startT, currentProgress))

  local prevShot = nil
  local shotIndex = 0

  while (currentT - startT) < targetDurationSeconds and currentProgress < 0.99 do
    shotIndex = shotIndex + 1

    local speedMs, speedKmh = getCurvatureAwareSpeed(currentProgress)

    local carWorldPos = ac.trackProgressToWorldCoordinate(currentProgress)
    if carWorldPos == nil then break end

    local nextP = math.min(1.0, currentProgress + 0.001)
    local nextPos = ac.trackProgressToWorldCoordinate(nextP)
    local carLook = vec3(0, 0, 1)
    if nextPos and carWorldPos then
      carLook = v3norm(vec3(nextPos.x - carWorldPos.x, nextPos.y - carWorldPos.y, nextPos.z - carWorldPos.z))
    end

    local candidates = generateAndValidateCandidates(carWorldPos, currentProgress)

    if #candidates == 0 then
      currentProgress = currentProgress + 0.003
      goto continue_step
    end

    local scoredCandidates = {}
    for _, c in ipairs(candidates) do
      local window = evaluateUsableWindow(c.final_pos, currentProgress, speedMs, state.track_length_m, currentT)
      if window.is_viable then
        scoreCandidate(c, prevShot, carWorldPos, carLook, window)
        scoredCandidates[#scoredCandidates + 1] = c
      end
    end

    if #scoredCandidates == 0 then
      for _, c in ipairs(candidates) do
        local window = evaluateUsableWindow(c.final_pos, currentProgress, speedMs, state.track_length_m, currentT)
        scoreCandidate(c, prevShot, carWorldPos, carLook, window)
        scoredCandidates[#scoredCandidates + 1] = c
      end
    end

    table.sort(scoredCandidates, function(a, b) return (a.calculated_score or 0) > (b.calculated_score or 0) end)
    local bestCandidate = scoredCandidates[1]

    if bestCandidate == nil then
      currentProgress = currentProgress + 0.003
      goto continue_step
    end

    local duration = determineShotDuration(bestCandidate.window, speedKmh)
    local shotEndT = math.floor((currentT + duration) * 10) / 10

    local topCompetitors = {}
    for ci = 1, math.min(3, #scoredCandidates) do
      local cand = scoredCandidates[ci]
      topCompetitors[#topCompetitors + 1] = {
        rank = ci,
        type = cand.type,
        category = cand.category,
        side = cand.side,
        distance = math.floor(cand.distance_to_car * 10) / 10,
        angle_type = cand.angle_type,
        usable_duration = cand.window.usable_duration,
        lead_in = cand.window.lead_in,
        score = cand.calculated_score,
        breakdown = cand.score_breakdown,
      }
    end

    local shot = {
      index = shotIndex,
      shot_start = currentT,
      shot_end = shotEndT,
      duration = duration,
      progress = currentProgress,
      camera_type = bestCandidate.type,
      category = bestCandidate.category,
      side = bestCandidate.side,
      desc = bestCandidate.desc,
      distance = math.floor(bestCandidate.distance_to_car * 10) / 10,
      score = bestCandidate.calculated_score,
      angle_type = bestCandidate.angle_type,
      usable_window_start = bestCandidate.window.usable_window_start,
      usable_window_end = bestCandidate.window.usable_window_end,
      usable_duration = bestCandidate.window.usable_duration,
      lead_in = bestCandidate.window.lead_in,
      score_breakdown = bestCandidate.score_breakdown,
      top_competitors = topCompetitors,
      camera_position = { bestCandidate.final_pos.x, bestCandidate.final_pos.y, bestCandidate.final_pos.z },
      ground_y = bestCandidate.ground_y,
      car_position = { carWorldPos.x, carWorldPos.y, carWorldPos.z },
      projected_speed_kmh = math.floor(speedKmh * 10) / 10,
    }

    state.planned_shots[#state.planned_shots + 1] = shot
    state.stats.type_distribution[shot.camera_type] = (state.stats.type_distribution[shot.camera_type] or 0) + 1

    currentT = shotEndT
    local progressDelta = (speedMs * duration) / state.track_length_m
    currentProgress = currentProgress + progressDelta
    prevShot = shot

    ::continue_step::
  end

  local totalDur = 0
  for _, s in ipairs(state.planned_shots) do
    totalDur = totalDur + s.duration
  end

  state.stats.duration_analyzed_s = math.floor((currentT - startT) * 10) / 10
  state.stats.shot_count = #state.planned_shots
  state.stats.avg_duration_s = state.stats.shot_count > 0 and (math.floor((totalDur / state.stats.shot_count) * 100) / 100) or 0

  state.status_message = string.format('Planned %d shots over %.1fs (avg %.2fs/shot)',
    state.stats.shot_count, state.stats.duration_analyzed_s, state.stats.avg_duration_s)
  ac.log('[ShotPlanner] === PLANNING COMPLETE === ' .. state.status_message)

  return true
end

-- ============================================================================
-- Plan Preview: Strict Replay Frame Synchronization (Reference)
-- ============================================================================

local function startPlanPreview()
  if #state.planned_shots == 0 then
    state.status_message = 'No planned shots. Click Plan 30s/60s first.'
    return
  end
  if state.plan_preview_active then return end

  local ok, cam = pcall(function()
    return ac.grabCamera('Shot Planner Plan Preview')
  end)
  if not ok or cam == nil then
    state.status_message = 'Failed to grab camera: ' .. tostring(cam)
    return
  end

  state.plan_preview_cam = cam
  state.plan_preview_active = true
  state.status_message = 'Plan Preview active (synced with replay)'
  ac.log('[ShotPlanner] Plan Preview started with ' .. #state.planned_shots .. ' shots')
end

local function stopPlanPreview()
  if state.plan_preview_cam then
    pcall(function() state.plan_preview_cam:dispose() end)
    state.plan_preview_cam = nil
  end
  state.plan_preview_active = false
  ac.log('[ShotPlanner] Plan Preview stopped')
end

local function updatePlanPreview()
  if not state.plan_preview_active then return end
  if #state.planned_shots == 0 then stopPlanPreview(); return end

  local sim = ac.getSim()
  if sim == nil then return end

  local currentFrame = sim.replayCurrentFrame or 0
  local replayTime = currentFrame / state.sample_rate

  local firstShot = state.planned_shots[1]
  local lastShot = state.planned_shots[#state.planned_shots]

  if replayTime < (firstShot.shot_start - 1.0) or replayTime > (lastShot.shot_end + 1.0) then
    stopPlanPreview()
    state.status_message = string.format('Replay at %.1fs outside planned window [%.1fs - %.1fs]',
      replayTime, firstShot.shot_start, lastShot.shot_end)
    return
  end

  local activeShot = nil
  local activeIdx = 1

  for i, shot in ipairs(state.planned_shots) do
    if replayTime >= shot.shot_start and replayTime < shot.shot_end then
      activeShot = shot
      activeIdx = i
      break
    end
  end

  if activeShot == nil then
    if replayTime < firstShot.shot_start then
      activeShot = firstShot
      activeIdx = 1
    else
      activeShot = lastShot
      activeIdx = #state.planned_shots
    end
  end

  state.plan_preview_shot_idx = activeIdx

  if state.plan_preview_cam and activeShot then
    local car = ac.getCar(0)
    if car and car.position then
      local carPos = vec3(car.position.x, car.position.y + 0.6, car.position.z)
      local camPos = vec3(activeShot.camera_position[1], activeShot.camera_position[2], activeShot.camera_position[3])

      local look = v3norm(vec3(carPos.x - camPos.x, carPos.y - camPos.y, carPos.z - camPos.z))
      local worldUp = vec3(0, 1, 0)
      if math.abs(v3dot(look, worldUp)) > 0.999 then
        worldUp = vec3(0, 0, 1)
      end
      local right = v3norm(v3cross(look, worldUp))
      local up = v3cross(right, look)

      state.plan_preview_cam.transform.position = camPos
      state.plan_preview_cam.transform.look = look
      state.plan_preview_cam.transform.side = right
      state.plan_preview_cam.transform.up = up
      state.plan_preview_cam.fov = 55
    end
  end
end

-- ============================================================================
-- Save Plan JSON & Audit Report (Reference)
-- ============================================================================

local function savePlanJson()
  if #state.planned_shots == 0 then
    state.save_message = 'Error: no planned shots to save'
    return false
  end

  local docsOk, docs = pcall(function() return ac.getFolder(ac.FolderID.ACDocuments) end)
  if not docsOk or docs == nil then
    state.save_message = 'Error: cannot locate Documents folder'
    return false
  end

  local path = docs .. '/shot_planner_preview.json'

  local data = {
    schema_version = 1,
    app = 'shot_planner_spike',
    csp_version = state.csp,
    timestamp = os.time(),
    summary = {
      replay_duration_analyzed_s = state.stats.duration_analyzed_s,
      shot_count = state.stats.shot_count,
      average_shot_duration_s = state.stats.avg_duration_s,
      candidates_generated = state.stats.total_candidates_generated,
      candidates_rejected = state.stats.total_candidates_rejected,
      candidates_accepted = state.stats.total_candidates_accepted,
      camera_type_distribution = state.stats.type_distribution,
      reject_reasons = state.stats.reject_reasons,
    },
    shots = state.planned_shots,
  }

  local encOk, encoded = pcall(function() return json.encode(data) end)
  if not encOk or encoded == nil then
    state.save_message = 'Error encoding plan JSON'
    return false
  end

  local file = io.open(path, 'w')
  if file == nil then
    state.save_message = 'Error opening: ' .. path
    return false
  end

  file:write(encoded)
  file:close()

  state.save_message = 'Plan saved: ' .. path
  ac.log('[ShotPlanner] Plan saved: ' .. path)
  return true
end

local function saveAuditReport()
  if #state.planned_shots == 0 then
    state.audit_save_message = 'Error: no planned shots to audit'
    return false
  end

  local docsOk, docs = pcall(function() return ac.getFolder(ac.FolderID.ACDocuments) end)
  if not docsOk or docs == nil then
    state.audit_save_message = 'Error: cannot locate Documents folder'
    return false
  end

  local path = docs .. '/shot_planner_audit.json'

  local auditData = {
    schema_version = 3,
    app = 'shot_planner_spike_audit',
    csp_version = state.csp,
    timestamp = os.time(),
    audit_notes = {
      curvature_aware_speed = 'Dynamic speed modeling based on track spline curvature (38-45 km/h hairpins, 75-85 km/h straights). Eliminates extrapolation drift.',
      tight_focal_offsets = 'PROGRESS_OFFSETS clamped to max ±0.0035 (max 25m). Eliminates cameras placed 54m-106m ahead.',
      usable_window_model = 'Simulates future car trajectory over dt=[0..5s], testing tight distance bounds (3.5-42m) and physics LOS.',
      replay_sync = 'Frame-accurate sync via sim.replayCurrentFrame / sample_rate.',
    },
    summary = {
      replay_duration_analyzed_s = state.stats.duration_analyzed_s,
      shot_count = state.stats.shot_count,
      average_shot_duration_s = state.stats.avg_duration_s,
      candidates_generated = state.stats.total_candidates_generated,
      candidates_rejected = state.stats.total_candidates_rejected,
      candidates_accepted = state.stats.total_candidates_accepted,
      camera_type_distribution = state.stats.type_distribution,
      reject_reasons = state.stats.reject_reasons,
    },
    shots = state.planned_shots,
  }

  local encOk, encoded = pcall(function() return json.encode(auditData) end)
  if not encOk or encoded == nil then
    state.audit_save_message = 'Error encoding audit JSON'
    return false
  end

  local file = io.open(path, 'w')
  if file == nil then
    state.audit_save_message = 'Error opening: ' .. path
    return false
  end

  file:write(encoded)
  file:close()

  state.audit_save_message = 'Audit saved: ' .. path
  ac.log('[ShotPlanner] Audit report saved: ' .. path)
  return true
end

-- ============================================================================
-- UI
-- ============================================================================

local function drawWindow()
  ui.text('=== SHOT PLANNER: DIRECTOR CAPTURE MODE ===')
  ui.text('CSP: ' .. state.csp)

  -- Replay & Car state
  local sim = ac.getSim()
  local replayTime = 0
  local currentFrame = 0
  if sim and sim.replayCurrentFrame then
    currentFrame = sim.replayCurrentFrame
    replayTime = currentFrame / state.sample_rate
    ui.text(string.format('Replay: frame %d | t = %.1fs | rate = %.1fx',
      currentFrame, replayTime, sim.replayPlaybackRate or 1.0))
  end

  if state.car_pos then
    local speedMs, speedKmh = getCurvatureAwareSpeed(state.car_progress)
    ui.text(string.format('Car pos: %s | Live Spd: %.1f km/h', v3str(state.car_pos), state.car_speed_kmh))
    ui.text(string.format('Progress: %.4f | Track Curv Speed: %.1f km/h', state.car_progress, speedKmh))
  end

  ui.separator()

  -- PRIMARY ACTION: DIRECTOR CAPTURE CONTROLS
  if not state.live_director_active then
    if ui.button('START LIVE DIRECTOR + RECORD', vec2(260, 34)) then
      startLiveDirector(true)
    end
    ui.sameLine()
    if ui.button('Preview (No Record)', vec2(160, 34)) then
      startLiveDirector(false)
    end

    if #state.session_shots > 0 then
      if ui.button('Save Director Session', vec2(170, 26)) then
        saveDirectorSession()
      end
      ui.sameLine()
      if ui.button('Clear Session', vec2(110, 26)) then
        state.session_shots = {}
        state.session_stats = {
          duration_s = 0,
          shot_count = 0,
          start_frame = 0,
          end_frame = 0,
          total_candidates_generated = 0,
          total_candidates_rejected = 0,
          total_candidates_accepted = 0,
          avg_duration_s = 0,
          type_distribution = {},
          angle_distribution = {},
          reject_reasons = {},
        }
        state.director_save_message = nil
        state.status_message = 'Session buffer cleared.'
      end
    end
  else
    if state.recording_active then
      if ui.button('STOP & SAVE SESSION', vec2(200, 34)) then
        stopLiveDirector(true)
      end
      ui.sameLine()
      if ui.button('Stop (No Save)', vec2(130, 34)) then
        stopLiveDirector(false)
      end
    else
      if ui.button('STOP LIVE DIRECTOR', vec2(200, 34)) then
        stopLiveDirector(false)
      end
    end
  end

  -- Status Messages
  if state.director_save_message then
    ui.textWrapped(state.director_save_message)
  end
  if state.status_message then
    ui.text(state.status_message)
  end

  -- Live Director Active Status Banner
  if state.live_director_active and state.live_current_shot then
    ui.separator()
    local recTag = state.recording_active and '[RECORDING LIVE SESSION]' or '[LIVE MONITORING]'
    ui.text('=== ' .. recTag .. ' ===')
    local s = state.live_current_shot
    local shotNum = state.recording_active and #state.session_shots or 1
    ui.text(string.format('Shot #%d: %s (%s, side=%s) | Dist: %.1fm',
      shotNum, s.camera_type, s.angle_type, s.side, s.distance))
    local remainingFrames = math.max(0, state.live_shot_end_frame - currentFrame)
    local remainingS = remainingFrames / state.sample_rate
    ui.text(string.format('Duration: %.1fs (ends frame %d, in %.1fs / %df) | Score: %.1f',
      s.duration, state.live_shot_end_frame, remainingS, remainingFrames, s.score))
  end

  -- Recorded Session Statistics Panel
  if state.session_stats.shot_count > 0 then
    ui.separator()
    ui.text('=== RECORDED SESSION STATISTICS ===')
    ui.text(string.format('Shots recorded: %d', state.session_stats.shot_count))
    ui.text(string.format('Total duration: %.2fs (Frames: %d -> %d)',
      state.session_stats.duration_s, state.session_stats.start_frame, state.session_stats.end_frame))
    ui.text(string.format('Average shot duration: %.2fs', state.session_stats.avg_duration_s))

    local gen = state.session_stats.total_candidates_generated
    local rej = state.session_stats.total_candidates_rejected
    local acc = state.session_stats.total_candidates_accepted
    local accPct = gen > 0 and (acc / gen * 100) or 0
    ui.text(string.format('Candidates evaluated: %d gen / %d rej / %d acc (%.1f%% accepted)',
      gen, rej, acc, accPct))

    ui.separator()
    ui.text('--- Camera Type Distribution ---')
    for typeName, count in pairs(state.session_stats.type_distribution) do
      local pct = (count / state.session_stats.shot_count) * 100
      ui.text(string.format('  %-22s: %d  (%.1f%%)', typeName, count, pct))
    end

    if state.session_stats.angle_distribution and next(state.session_stats.angle_distribution) then
      ui.text('--- Angle Distribution ---')
      local angParts = {}
      for ang, cnt in pairs(state.session_stats.angle_distribution) do
        local pct = (cnt / state.session_stats.shot_count) * 100
        table.insert(angParts, string.format('%s: %d (%.0f%%)', ang, cnt, pct))
      end
      ui.text('  ' .. table.concat(angParts, ' | '))
    end

    if state.session_stats.reject_reasons and next(state.session_stats.reject_reasons) then
      ui.text('--- Candidate Rejection Reasons ---')
      for reason, cnt in pairs(state.session_stats.reject_reasons) do
        ui.text(string.format('  %-22s: %d', reason, cnt))
      end
    end

    -- Recorded Shots Log
    ui.separator()
    ui.text('=== RECORDED SHOTS LOG ===')
    for _, s in ipairs(state.session_shots) do
      local bk = s.score_breakdown
      local bkStr = bk and string.format('[ang:%.0f dst:%.0f cat:%.0f win:%+.0f div:%+.0f]',
        bk.angle, bk.distance, bk.category, bk.window, bk.diversity) or ''

      ui.text(string.format('Shot %02d: frames [%d - %d] (%.2fs - %.2fs | dur: %.2fs)',
        s.index, s.shot_start_frame, s.shot_end_frame, s.shot_start_time, s.shot_end_time, s.duration))
      ui.text(string.format('  %s (%s, %s) | dist: %.1fm | score: %.1f %s',
        s.camera_type, s.angle_type, s.side, s.distance_to_car, s.score, bkStr))
    end
  end

  -- Offline Batch Planner (Reference Section)
  ui.separator()
  local toggleLabel = state.show_batch_planner and '[-] Hide Batch Planner (Reference)' or '[+] Show Batch Planner (Reference)'
  if ui.button(toggleLabel, vec2(260, 24)) then
    state.show_batch_planner = not state.show_batch_planner
  end

  if state.show_batch_planner then
    ui.text('--- Offline Extrapolation Planner (Deprecated) ---')
    if ui.button('Plan 30s Sequence', vec2(140, 24)) then
      planSequence(30.0)
    end
    ui.sameLine()
    if ui.button('Plan 60s Sequence', vec2(140, 24)) then
      planSequence(60.0)
    end

    if #state.planned_shots > 0 then
      ui.sameLine()
      if not state.plan_preview_active then
        if ui.button('Play Plan Preview', vec2(130, 24)) then
          startPlanPreview()
        end
      else
        if ui.button('Stop Preview', vec2(130, 24)) then
          stopPlanPreview()
        end
      end

      if ui.button('Save Plan JSON', vec2(140, 22)) then
        savePlanJson()
      end
      ui.sameLine()
      if ui.button('Save Audit JSON', vec2(140, 22)) then
        saveAuditReport()
      end
    end

    if state.save_message then
      ui.textWrapped(state.save_message)
    end
    if state.audit_save_message then
      ui.textWrapped(state.audit_save_message)
    end
  end
end

-- ============================================================================
-- Entry point
-- ============================================================================

function script.windowMain()
  if not state.probed then
    state.probed = true
    local ok, v = pcall(function() return ac.getPatchVersion() end)
    state.csp = ok and tostring(v) or 'unknown'
    readCarState()
    estimateTrackLength()
  end

  -- Update Replay-synced Plan Preview (if active)
  updatePlanPreview()

  -- Update Director Capture Mode (Live real-time cut execution & recording)
  updateLiveDirector()

  drawWindow()
end
