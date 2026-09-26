-- Camera Placement Spike — CSP 0.3.0-preview595
--
-- Tests generation and validation of camera placement candidates.
-- Uses only APIs confirmed in runtime:
--   ac.trackProgressToWorldCoordinate(progress) -> vec3
--   ac.getTrackAISplineSides(progress) -> vec2 (left_m, right_m)
--   ac.trackCoordinateToWorld(vec3(lateral, height, progress)) -> vec3
--   ac.worldCoordinateToTrack(vec3) -> vec3 (lateral, height, progress)
--   render.createRay + ray:physics(outPos, outNormal) -> distance or -1
--   ac.getCar(0).position / .look / .splinePosition
--
-- Does NOT modify timeline_driver_spike or aidirector_spike.

-- ============================================================================
-- Configuration
-- ============================================================================

local RAISE_HEIGHT = 80.0    -- metres above point to cast ray from
local RAY_LENGTH = 200.0     -- metres for downward raycast
local MIN_CLEARANCE = 0.5    -- minimum metres above ground
local LOS_TOLERANCE = 2.0    -- metres tolerance for LOS check

-- Camera type definitions: { name, lateral_offset_factor, height_above_ground, description }
-- lateral_offset_factor: relative to track edge (1.0 = edge of asphalt, >1.0 = beyond edge).
-- Calibrated for mountain pass: ~0.5m-1.0m beyond guardrail to avoid foliage/bush occlusion.
local CAMERA_TYPES = {
  -- Roadside cameras at eye level, tight to track edge (approx 0.5 - 1.0m beyond edge/guardrail)
  { name = 'roadside_left_close',   lateral = -1.25, height = 1.4, desc = 'Left edge, eye level (~0.8m off track)' },
  { name = 'roadside_right_close',  lateral =  1.25, height = 1.4, desc = 'Right edge, eye level (~0.8m off track)' },

  -- Moderated wide (fallback / spectator berm, ~2.2m off track instead of old 8.7m in forest)
  { name = 'roadside_left_wide',    lateral = -1.65, height = 1.6, desc = 'Wide left berm (~2.2m off track)' },
  { name = 'roadside_right_wide',   lateral =  1.65, height = 1.6, desc = 'Wide right berm (~2.2m off track)' },

  -- Low cameras (dramatic angle, tight to edge, height 0.6m above min clearance)
  { name = 'low_left',              lateral = -1.15, height = 0.6, desc = 'Low left edge, dramatic angle' },
  { name = 'low_right',             lateral =  1.15, height = 0.6, desc = 'Low right edge, dramatic angle' },

  -- Elevated cameras (testing multiple heights to look OVER roadside vegetation/bushes)
  { name = 'elevated_left_low',     lateral = -1.35, height = 2.8, desc = 'Elevated left, over bushes (2.8m)' },
  { name = 'elevated_left_mid',     lateral = -1.45, height = 4.5, desc = 'Elevated left boom pole (4.5m)' },
  { name = 'elevated_right',        lateral =  1.30, height = 3.5, desc = 'Elevated right overview (3.5m)' },
  { name = 'elevated_high',         lateral = -0.80, height = 7.5, desc = 'High crane / overhead (7.5m)' },
}

-- Progress offsets: where to place cameras relative to car's current progress.
-- Main range for tight mountain pass: ±0.003 ... ±0.007 (approx 20m - 55m).
-- Immediate pass-by: 0.0 (approx 3m - 10m).
-- Distant telephoto: ±0.015 (approx 110-115m), kept as a rare candidate.
local PROGRESS_OFFSETS = { -0.015, -0.007, -0.003, 0.0, 0.003, 0.007, 0.015 }

-- ============================================================================
-- State
-- ============================================================================

local state = {
  csp = 'unknown',
  probed = false,
  -- Car info
  car_pos = nil,
  car_look = nil,
  car_progress = nil,    -- track spline progress [0..1]
  car_progress_method = 'unknown',
  -- Results
  candidates = {},       -- array of candidate results
  summary = nil,         -- { total, accepted, rejected, reasons }
  error = nil,
  running = false,
  done = false,
  -- Preview mode
  preview_active = false,
  preview_cam = nil,         -- GrabbedCamera
  preview_list = {},         -- filtered accepted candidates
  preview_idx = 0,           -- current index in preview_list
  preview_start = 0,         -- os.clock() when current preview started
  preview_hold = 3.0,        -- seconds per candidate
  preview_frame = 0,         -- total frames in preview
  preview_paused = false,    -- pause auto-advance
  save_status = nil,         -- JSON save status message
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

local function v3cross(a, b)
  return vec3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  )
end

-- ============================================================================
-- Terrain queries (proven APIs)
-- ============================================================================

--- Raycast straight down from a point to find ground height.
--- Returns (ground_y, hit_distance) or (nil, nil) if no hit.
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

  -- Ground Y = origin.y - distance
  local groundY = origin.y - dist
  -- Or use outPos if it was written to
  if outPos.y ~= 9e9 then
    groundY = outPos.y
  end

  return groundY, dist
end

--- Raycast from camera position toward target to check line of sight.
--- Returns (clear, hit_distance, total_distance).
--- clear=true means no obstruction between camera and target.
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
    -- No hit = clear line of sight
    return true, -1, totalDist
  end

  -- Hit something. Is it beyond the target?
  if dist >= totalDist - LOS_TOLERANCE then
    return true, dist, totalDist  -- hit is past the target
  end

  return false, dist, totalDist  -- obstruction between camera and target
end

-- ============================================================================
-- Car state reading
-- ============================================================================

local function readCarState()
  local ok, car = pcall(function() return ac.getCar(0) end)
  if not ok or car == nil then
    state.error = 'Cannot read car state'
    return false
  end

  state.car_pos = vec3(car.position.x, car.position.y, car.position.z)

  -- Car look direction
  local lookOk, carLook = pcall(function() return car.look end)
  if lookOk and carLook ~= nil then
    state.car_look = vec3(carLook.x, carLook.y, carLook.z)
  else
    state.car_look = vec3(0, 0, 1)
  end

  -- Try to get track progress via car.splinePosition
  local spOk, spVal = pcall(function() return car.splinePosition end)
  if spOk and spVal ~= nil and type(spVal) == 'number' then
    state.car_progress = spVal
    state.car_progress_method = 'car.splinePosition'
    return true
  end

  -- Fallback: use ac.worldCoordinateToTrack
  local wtOk, trackCoord = pcall(function()
    return ac.worldCoordinateToTrack(car.position)
  end)
  if wtOk and trackCoord ~= nil then
    state.car_progress = trackCoord.z  -- z = progress
    state.car_progress_method = 'ac.worldCoordinateToTrack(.z)'
    return true
  end

  state.error = 'Cannot determine car track progress'
  return false
end

-- ============================================================================
-- Candidate generation and validation
-- ============================================================================

local function generateCandidates()
  state.candidates = {}
  state.summary = nil
  state.error = nil
  state.done = false

  if not readCarState() then return end

  local carPos = state.car_pos
  local carProgress = state.car_progress

  ac.log('[Placement] === GENERATING CANDIDATES ===')
  ac.log(string.format('[Placement] car_pos=%s progress=%.4f method=%s',
    v3str(carPos), carProgress, state.car_progress_method))
  ac.log(string.format('[Placement] car_look=%s', v3str(state.car_look)))

  local total = 0
  local accepted = 0
  local rejected = 0
  local reasons = {}  -- { reason_string = count }

  for _, progOffset in ipairs(PROGRESS_OFFSETS) do
    local progress = carProgress + progOffset
    -- Wrap progress to [0, 1]
    if progress < 0 then progress = progress + 1.0 end
    if progress > 1 then progress = progress - 1.0 end

    -- Get track centre and width at this progress
    local centreOk, centre = pcall(function()
      return ac.trackProgressToWorldCoordinate(progress)
    end)
    if not centreOk or centre == nil then
      ac.log(string.format('[Placement] Skip progress=%.4f: trackProgressToWorldCoordinate failed', progress))
      goto continue_progress
    end

    local sidesOk, sides = pcall(function()
      return ac.getTrackAISplineSides(progress)
    end)
    local leftWidth = (sidesOk and sides ~= nil) and sides.x or 3.0
    local rightWidth = (sidesOk and sides ~= nil) and sides.y or 3.0

    for _, camType in ipairs(CAMERA_TYPES) do
      total = total + 1

      local candidate = {
        type = camType.name,
        desc = camType.desc,
        progress = progress,
        prog_offset = progOffset,
        lateral = camType.lateral,
        target_height = camType.height,
        status = 'pending',
        reject_reason = nil,
        world_pos = nil,
        ground_y = nil,
        final_pos = nil,
        distance_to_car = nil,
        los_clear = nil,
        los_hit_dist = nil,
      }

      -- Calculate lateral offset in metres
      -- lateral factor * half-width gives world metres from centre
      local halfWidth = camType.lateral > 0 and rightWidth or leftWidth
      local lateralMetres = camType.lateral * halfWidth

      -- Get world position via track coordinates
      -- trackCoordinateToWorld: x = normalised lateral (-1 to +1 for track bounds),
      -- but we want metres beyond the track, so we compute raw position
      local rawPos = vec3(
        centre.x + lateralMetres * 0.5,  -- rough lateral offset
        centre.y,
        centre.z
      )

      -- Better approach: use track coordinate system
      -- x in trackCoordinateToWorld is -1 (left edge) to +1 (right edge)
      -- We want beyond the edge, so we use larger values
      local trackLateral = camType.lateral  -- already a factor
      local twOk, worldPos = pcall(function()
        return ac.trackCoordinateToWorld(vec3(trackLateral, 0, progress))
      end)

      if not twOk or worldPos == nil then
        candidate.status = 'rejected'
        candidate.reject_reason = 'trackCoordinateToWorld failed'
        reasons[candidate.reject_reason] = (reasons[candidate.reject_reason] or 0) + 1
        rejected = rejected + 1
        state.candidates[#state.candidates + 1] = candidate
        goto continue_type
      end

      candidate.world_pos = vec3(worldPos.x, worldPos.y, worldPos.z)

      -- Find ground height at this position
      local groundY, hitDist = findGround(worldPos)

      if groundY == nil then
        candidate.status = 'rejected'
        candidate.reject_reason = 'no ground (raycast miss)'
        reasons[candidate.reject_reason] = (reasons[candidate.reject_reason] or 0) + 1
        rejected = rejected + 1
        state.candidates[#state.candidates + 1] = candidate
        goto continue_type
      end

      candidate.ground_y = groundY

      -- Place camera at ground + desired height
      local camY = groundY + camType.height
      local camPos = vec3(worldPos.x, camY, worldPos.z)
      candidate.final_pos = camPos

      -- Check minimum clearance
      if camY < groundY + MIN_CLEARANCE then
        candidate.status = 'rejected'
        candidate.reject_reason = string.format('below min clearance (%.1fm)', camY - groundY)
        reasons[candidate.reject_reason] = (reasons[candidate.reject_reason] or 0) + 1
        rejected = rejected + 1
        state.candidates[#state.candidates + 1] = candidate
        goto continue_type
      end

      -- Distance to car
      candidate.distance_to_car = v3dist(camPos, carPos)

      -- Check distance is reasonable (not too far, not too close)
      if candidate.distance_to_car > 200 then
        candidate.status = 'rejected'
        candidate.reject_reason = string.format('too far (%.0fm)', candidate.distance_to_car)
        reasons[candidate.reject_reason] = (reasons[candidate.reject_reason] or 0) + 1
        rejected = rejected + 1
        state.candidates[#state.candidates + 1] = candidate
        goto continue_type
      end
      if candidate.distance_to_car < 2 then
        candidate.status = 'rejected'
        candidate.reject_reason = string.format('too close (%.1fm)', candidate.distance_to_car)
        reasons[candidate.reject_reason] = (reasons[candidate.reject_reason] or 0) + 1
        rejected = rejected + 1
        state.candidates[#state.candidates + 1] = candidate
        goto continue_type
      end

      -- Check line of sight to car
      local losClear, losHitDist, losTotalDist = checkLineOfSight(camPos, carPos)
      candidate.los_clear = losClear
      candidate.los_hit_dist = losHitDist

      if not losClear then
        candidate.status = 'rejected'
        candidate.reject_reason = string.format('LOS blocked at %.1fm (car at %.1fm)',
          losHitDist, losTotalDist)
        reasons[candidate.reject_reason] = (reasons[candidate.reject_reason] or 0) + 1
        rejected = rejected + 1
        state.candidates[#state.candidates + 1] = candidate
        goto continue_type
      end

      -- All checks passed
      candidate.status = 'accepted'
      accepted = accepted + 1
      state.candidates[#state.candidates + 1] = candidate

      ::continue_type::
    end

    ::continue_progress::
  end

  -- Build summary
  state.summary = {
    total = total,
    accepted = accepted,
    rejected = rejected,
    reasons = reasons,
    car_pos = v3str(carPos),
    car_progress = carProgress,
    car_progress_method = state.car_progress_method,
  }

  -- Log summary
  ac.log('[Placement] === SUMMARY ===')
  ac.log(string.format('[Placement] Total: %d, Accepted: %d, Rejected: %d',
    total, accepted, rejected))
  for reason, count in pairs(reasons) do
    ac.log(string.format('[Placement]   reject: %s (%d)', reason, count))
  end

  -- Log first 5 accepted candidates
  local showCount = 0
  for _, c in ipairs(state.candidates) do
    if c.status == 'accepted' and showCount < 5 then
      showCount = showCount + 1
      ac.log(string.format('[Placement] ACCEPTED: %s at %s ground=%.1f dist=%.1fm prog_off=%.3f',
        c.type, v3str(c.final_pos), c.ground_y, c.distance_to_car, c.prog_offset))
    end
  end

  state.done = true
  state.running = false
end

-- ============================================================================
-- Preview mode: cycle through accepted candidates with live camera
-- ============================================================================

local function buildPreviewList()
  state.preview_list = {}
  for _, c in ipairs(state.candidates) do
    if c.status == 'accepted' and c.final_pos then
      state.preview_list[#state.preview_list + 1] = c
    end
  end
end

local function applyPreviewCamera(candidate)
  if state.preview_cam == nil then return end

  local camPos = candidate.final_pos

  -- Read the car's LIVE position (not the one from generation time)
  local car = ac.getCar(0)
  if car == nil then return end
  local carPos = vec3(car.position.x, car.position.y, car.position.z)

  -- Compute look direction toward the car's current position
  local look = v3norm(vec3(
    carPos.x - camPos.x,
    carPos.y - camPos.y,
    carPos.z - camPos.z
  ))

  -- Full orthonormal basis (proven in camera_look_spike)
  local worldUp = vec3(0, 1, 0)
  local dotUp = math.abs(look.x * worldUp.x + look.y * worldUp.y + look.z * worldUp.z)
  if dotUp > 0.999 then
    worldUp = vec3(0, 0, 1)
  end
  local right = v3norm(v3cross(look, worldUp))
  local up = v3cross(right, look)

  state.preview_cam.transform.position = camPos
  state.preview_cam.transform.look = look
  state.preview_cam.transform.side = right
  state.preview_cam.transform.up = up
  state.preview_cam.fov = 55
end

local function savePreviewReport()
  local jsonOk, json = pcall(function() return require('json') end)
  if not jsonOk or json == nil then
    state.save_status = 'Error: json library not found'
    return false
  end

  local acceptedList = {}
  for _, c in ipairs(state.candidates) do
    if c.status == 'accepted' and c.final_pos then
      acceptedList[#acceptedList + 1] = {
        index = #acceptedList + 1,
        type = c.type,
        desc = c.desc,
        progress = c.progress,
        prog_offset = c.prog_offset,
        camera_position = { c.final_pos.x, c.final_pos.y, c.final_pos.z },
        ground_y = c.ground_y,
        car_position = state.car_pos and { state.car_pos.x, state.car_pos.y, state.car_pos.z } or nil,
        distance = c.distance_to_car,
        height = c.target_height,
        los_clear = c.los_clear,
        validation_result = c.status,
      }
    end
  end

  local docsOk, docs = pcall(function() return ac.getFolder(ac.FolderID.ACDocuments) end)
  if not docsOk or docs == nil then
    state.save_status = 'Error: cannot get Documents folder'
    return false
  end

  local path = docs .. '/camera_placement_preview.json'

  local report = {
    schema_version = 1,
    app = 'camera_placement_spike',
    csp_version = state.csp,
    car = {
      position = state.car_pos and { state.car_pos.x, state.car_pos.y, state.car_pos.z } or nil,
      progress = state.car_progress,
      progress_method = state.car_progress_method,
    },
    summary = state.summary,
    accepted_count = #acceptedList,
    accepted_candidates = acceptedList,
  }

  local encOk, encoded = pcall(function() return json.encode(report) end)
  if not encOk or encoded == nil then
    state.save_status = 'Error encoding JSON'
    return false
  end

  local file = io.open(path, 'w')
  if file == nil then
    state.save_status = 'Error opening: ' .. path
    return false
  end

  file:write(encoded)
  file:close()

  state.save_status = 'Report saved: ' .. path
  ac.log('[Placement] Report saved: ' .. path)
  return true
end

local function startPreview()
  if state.preview_active then return end

  buildPreviewList()
  if #state.preview_list == 0 then
    state.error = 'No accepted candidates to preview. Generate candidates first.'
    return
  end

  local ok, cam = pcall(function()
    return ac.grabCamera('Camera Placement Preview')
  end)
  if not ok or cam == nil then
    state.error = 'Failed to grab camera: ' .. tostring(cam)
    return
  end

  state.preview_cam = cam
  state.preview_active = true
  state.preview_paused = false
  state.preview_idx = 1
  state.preview_start = os.clock()
  state.preview_frame = 0
  state.error = nil

  ac.log(string.format('[Placement] Preview started: %d candidates', #state.preview_list))
end

local function stopPreview()
  if state.preview_cam then
    pcall(function() state.preview_cam:dispose() end)
    state.preview_cam = nil
  end
  state.preview_active = false
  ac.log('[Placement] Preview stopped')
end

local function previewNext()
  if not state.preview_active then return end
  state.preview_idx = state.preview_idx + 1
  if state.preview_idx > #state.preview_list then
    state.preview_idx = 1
  end
  state.preview_start = os.clock()
  ac.log(string.format('[Placement] Preview -> %d/%d: %s',
    state.preview_idx, #state.preview_list,
    state.preview_list[state.preview_idx].type))
end

local function previewPrev()
  if not state.preview_active then return end
  state.preview_idx = state.preview_idx - 1
  if state.preview_idx < 1 then
    state.preview_idx = #state.preview_list
  end
  state.preview_start = os.clock()
  ac.log(string.format('[Placement] Preview -> %d/%d: %s',
    state.preview_idx, #state.preview_list,
    state.preview_list[state.preview_idx].type))
end

local function previewSelect(idx)
  if not state.preview_active then
    startPreview()
  end
  if state.preview_active and idx >= 1 and idx <= #state.preview_list then
    state.preview_idx = idx
    state.preview_start = os.clock()
    state.preview_paused = true
  end
end

local function updatePreview()
  if not state.preview_active then return end
  if #state.preview_list == 0 then stopPreview(); return end

  state.preview_frame = state.preview_frame + 1

  -- Auto-advance after hold time if not paused
  if not state.preview_paused then
    local elapsed = os.clock() - state.preview_start
    if elapsed >= state.preview_hold then
      local nextIdx = state.preview_idx + 1
      if nextIdx > #state.preview_list then
        -- Cycled through all — stop and release camera
        ac.log('[Placement] Preview complete: all candidates shown')
        stopPreview()
        return
      end
      state.preview_idx = nextIdx
      state.preview_start = os.clock()
    end
  end

  -- Apply camera each frame (tracks car movement)
  local candidate = state.preview_list[state.preview_idx]
  if candidate then
    applyPreviewCamera(candidate)
  end
end

-- ============================================================================
-- UI
-- ============================================================================

local function drawWindow()
  ui.text('=== CAMERA PLACEMENT SPIKE ===')
  ui.text('CSP: ' .. state.csp)

  if state.error then
    ui.textWrapped('ERROR: ' .. state.error)
    ui.separator()
  end

  -- Car state
  if state.car_pos then
    ui.text(string.format('Car: %s', v3str(state.car_pos)))
    ui.text(string.format('Look: %s', v3str(state.car_look)))
    ui.text(string.format('Progress: %.4f (%s)', state.car_progress or 0, state.car_progress_method))
  else
    ui.text('Car: not read yet')
  end

  ui.separator()

  -- Controls: Generation
  if not state.running then
    if ui.button('Generate Candidates', vec2(180, 30)) then
      state.running = true
      generateCandidates()
    end
    ui.sameLine()
    if ui.button('Refresh Car Pos', vec2(140, 30)) then
      readCarState()
    end
  else
    ui.text('Generating candidates...')
  end

  -- Preview & Report controls
  if state.summary and state.summary.accepted > 0 then
    ui.separator()
    if not state.preview_active then
      if ui.button(string.format('Preview Accepted (%d)', state.summary.accepted), vec2(200, 30)) then
        startPreview()
      end
      ui.sameLine()
      if ui.button('Save Preview Report', vec2(160, 30)) then
        savePreviewReport()
      end
    end

    if state.save_status then
      ui.textWrapped(state.save_status)
    end
  end

  -- Live Preview Panel
  if state.preview_active and #state.preview_list > 0 then
    local c = state.preview_list[state.preview_idx]
    local elapsed = os.clock() - state.preview_start
    local remaining = math.max(0, state.preview_hold - elapsed)

    ui.separator()
    ui.text('=== PREVIEW ACTIVE ===')
    ui.text(string.format('Candidate: %d / %d', state.preview_idx, #state.preview_list))
    if c then
      ui.text(string.format('Type: %s', c.type))
      ui.text(string.format('Desc: %s', c.desc or ''))
      ui.text(string.format('Progress: %.4f (offset: %+.3f)', c.progress, c.prog_offset))
      ui.text(string.format('Camera pos: %s', v3str(c.final_pos)))
      ui.text(string.format('Ground height: %.1fm', c.ground_y or 0))
      ui.text(string.format('Height above ground: %.1fm', c.target_height or 0))
      ui.text(string.format('Distance to car: %.1fm', c.distance_to_car or 0))
      ui.text(string.format('LOS: %s (hit=%.1fm)', c.los_clear and 'CLEAR' or 'BLOCKED', c.los_hit_dist or -1))
    end

    if state.preview_paused then
      ui.text('Timer: [PAUSED]')
    else
      ui.text(string.format('Next candidate in: %.1fs', remaining))
    end

    -- Preview Navigation Controls
    if ui.button('<< Prev', vec2(75, 26)) then
      previewPrev()
    end
    ui.sameLine()
    local pauseLabel = state.preview_paused and 'Resume' or 'Pause'
    if ui.button(pauseLabel, vec2(80, 26)) then
      state.preview_paused = not state.preview_paused
      state.preview_start = os.clock()
    end
    ui.sameLine()
    if ui.button('Next >>', vec2(75, 26)) then
      previewNext()
    end
    ui.sameLine()
    if ui.button('Stop Preview', vec2(110, 26)) then
      stopPreview()
    end
    ui.separator()
  end

  -- Summary
  if state.summary then
    ui.separator()
    ui.text('=== SUMMARY ===')
    ui.text(string.format('Total candidates: %d', state.summary.total))
    ui.text(string.format('Accepted: %d', state.summary.accepted))
    ui.text(string.format('Rejected: %d', state.summary.rejected))

    ui.separator()
    ui.text('=== REJECT REASONS ===')
    for reason, count in pairs(state.summary.reasons) do
      ui.text(string.format('  %dx %s', count, reason))
    end

    -- Show accepted candidates
    ui.separator()
    ui.text('=== ACCEPTED CANDIDATES ===')
    local showCount = 0
    for i, c in ipairs(state.candidates) do
      if c.status == 'accepted' then
        showCount = showCount + 1
        if showCount <= 12 then
          ui.text(string.format('#%d: %s (dist=%.0fm, off=%+.3f)',
            showCount, c.type, c.distance_to_car or 0, c.prog_offset))
          ui.sameLine()
          if ui.button(string.format('View##%d', showCount), vec2(50, 18)) then
            buildPreviewList()
            previewSelect(showCount)
          end
        end
      end
    end
    if showCount > 12 then
      ui.text(string.format('  ... and %d more', showCount - 12))
    end

    -- Show some rejected candidates
    ui.separator()
    ui.text('=== REJECTED SAMPLES ===')
    local rejCount = 0
    for _, c in ipairs(state.candidates) do
      if c.status == 'rejected' then
        rejCount = rejCount + 1
        if rejCount <= 6 then
          ui.text(string.format('%s: %s', c.type, c.reject_reason))
          if c.world_pos then
            ui.text(string.format('  raw_pos=%s', v3str(c.world_pos)))
          end
        end
      end
    end
    if rejCount > 6 then
      ui.text(string.format('  ... and %d more', rejCount - 6))
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
  end

  -- Update camera preview every frame
  updatePreview()

  drawWindow()
end

