-- Timeline Driver Spike — CSP 0.3.0-preview595
--
-- First MVP of the realtime camera driver. Reads timeline.json produced by
-- `aidirector direct` and drives the camera during replay playback.
--
-- Uses ONLY APIs confirmed working in runtime on CSP 0.3.0-preview595:
--   ac.grabCamera / cam:dispose / cam.transform.position/look/side/up / cam.fov
--   ac.getFolder(ac.FolderID.ACDocuments)
--   sim.replayCurrentFrame / sim.replayFrames / sim.replayPlaybackRate
--   io.open / json.decode
--
-- Timeline JSON format (schema_version 1, from aidirector direct):
--   .session.sample_rate_hz  : float — replay sample rate
--   .shots[]                 : array of shot objects
--     .start_s / .end_s      : float — shot time boundaries
--     .rig                   : string — "roadside", "follow", etc.
--     .camera.keyframes[]    : array of keyframe objects
--       .t                   : float — time in seconds from session start
--       .position            : [x, y, z]
--       .look                : [x, y, z] — DIRECTION vector (unit)
--       .up                  : [x, y, z] — up vector (unit)
--       .fov                 : float — vertical degrees
--       .own_share           : float [0..1] — blend weight
--       .dof_distance        : float (optional)
--       .dof_factor          : float (optional)
--
-- Camera math (proven in camera_look_spike):
--   forward = look direction (positive = viewing direction, NOT -Z)
--   right   = normalize(cross(forward, worldUp))
--   up      = cross(right, forward)
--   cam.transform.position/look/side/up must all be set each frame

local json = require('json')

-- ============================================================================
-- State
-- ============================================================================

local state = {
  csp = 'unknown',
  probed = false,
  -- Timeline data
  timeline = nil,          -- parsed JSON object
  shots = nil,             -- shots array
  sample_rate = 33.3333,   -- from timeline.session.sample_rate_hz
  -- Playback
  cam = nil,               -- GrabbedCamera
  active = false,          -- currently driving the camera
  current_shot_idx = -1,   -- index into shots array (0-based in data, 1-based in lua table)
  -- Info for UI
  error = nil,
  status = 'idle',
  last_t = 0,
  total_frames = 0,
  timeline_file = '',
}

-- ============================================================================
-- Vector math (no external dependencies)
-- ============================================================================

local function v3len(v)
  return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
end

local function v3norm(v)
  local len = v3len(v)
  if len < 1e-12 then return vec3(0, 0, 1) end
  return vec3(v.x / len, v.y / len, v.z / len)
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

-- ============================================================================
-- Timeline loading
-- ============================================================================

local function loadTimeline()
  local docs = ac.getFolder(ac.FolderID.ACDocuments)
  local path = docs .. '/aidirector/timeline.json'
  state.timeline_file = path

  ac.log('[Timeline Driver] Loading: ' .. path)

  local file = io.open(path, 'r')
  if file == nil then
    state.error = 'Cannot open: ' .. path
    return false
  end

  local content = file:read('*a')
  file:close()

  if content == nil or #content == 0 then
    state.error = 'File is empty: ' .. path
    return false
  end

  local ok, data = pcall(function() return json.decode(content) end)
  if not ok or data == nil then
    state.error = 'JSON parse failed: ' .. tostring(data)
    return false
  end

  -- Validate schema version
  if data.schema_version ~= 1 then
    state.error = 'Unsupported schema_version: ' .. tostring(data.schema_version)
    return false
  end

  -- Extract sample rate
  if data.session and data.session.sample_rate_hz then
    state.sample_rate = data.session.sample_rate_hz
  end

  -- Extract shots
  if data.shots == nil or #data.shots == 0 then
    state.error = 'Timeline has no shots'
    return false
  end

  state.timeline = data
  state.shots = data.shots

  ac.log(string.format('[Timeline Driver] Loaded: %d shots, %.1fs, sample_rate=%.4f Hz',
    #state.shots, data.summary and data.summary.duration_s or 0, state.sample_rate))

  return true
end

-- ============================================================================
-- Shot and keyframe lookup
-- ============================================================================

--- Find the shot covering time t. Returns shot table (1-indexed in lua) or nil.
local function findShot(t)
  -- Linear scan (15 shots is fast enough; binary search is overkill for MVP)
  for i, shot in ipairs(state.shots) do
    if t >= shot.start_s and t < shot.end_s then
      return shot, i
    end
  end
  -- Check if t equals the very end of the last shot
  local last = state.shots[#state.shots]
  if last and math.abs(t - last.end_s) < 0.001 then
    return last, #state.shots
  end
  return nil, -1
end

--- Find the two keyframes bracketing time t, and the interpolation alpha.
--- Returns kf_a, kf_b, alpha (0..1).
local function findKeyframes(shot, t)
  local kfs = shot.camera.keyframes
  if kfs == nil or #kfs == 0 then return nil, nil, 0 end

  -- Clamp to shot boundaries
  if t <= kfs[1].t then return kfs[1], kfs[1], 0 end
  if t >= kfs[#kfs].t then return kfs[#kfs], kfs[#kfs], 0 end

  -- Linear scan for the bracket
  for i = 1, #kfs - 1 do
    if t >= kfs[i].t and t < kfs[i + 1].t then
      local span = kfs[i + 1].t - kfs[i].t
      local alpha = 0
      if span > 1e-9 then
        alpha = (t - kfs[i].t) / span
      end
      return kfs[i], kfs[i + 1], alpha
    end
  end

  -- Fallback
  return kfs[#kfs], kfs[#kfs], 0
end

--- Extract a vec3 from a JSON [x,y,z] array.
local function toVec3(arr)
  if arr == nil or #arr < 3 then return vec3(0, 0, 0) end
  return vec3(arr[1], arr[2], arr[3])
end

--- Interpolate between two keyframes at alpha, apply to camera.
local function applyKeyframe(kf_a, kf_b, alpha)
  local cam = state.cam
  if cam == nil then return end

  -- Interpolate position
  local pos_a = toVec3(kf_a.position)
  local pos_b = toVec3(kf_b.position)
  local pos = v3lerp(pos_a, pos_b, alpha)

  -- Interpolate look direction (normalised after lerp)
  local look_a = toVec3(kf_a.look)
  local look_b = toVec3(kf_b.look)
  local look = v3norm(v3lerp(look_a, look_b, alpha))

  -- Compute full orthonormal basis (proven in camera_look_spike)
  local worldUp = vec3(0, 1, 0)

  -- Handle degenerate case: look nearly parallel to worldUp
  local dotUp = math.abs(look.x * worldUp.x + look.y * worldUp.y + look.z * worldUp.z)
  if dotUp > 0.999 then
    -- Looking straight up or down — use the up vector from the keyframe data
    local up_a = toVec3(kf_a.up)
    local up_b = toVec3(kf_b.up)
    worldUp = v3norm(v3lerp(up_a, up_b, alpha))
  end

  local right = v3norm(v3cross(look, worldUp))
  local up = v3cross(right, look)

  -- Apply to camera
  cam.transform.position = pos
  cam.transform.look = look
  cam.transform.side = right
  cam.transform.up = up

  -- Interpolate FOV
  local fov = kf_a.fov + (kf_b.fov - kf_a.fov) * alpha
  cam.fov = fov

  -- Interpolate own_share
  local own_a = kf_a.own_share or 1
  local own_b = kf_b.own_share or 1
  cam.ownShare = own_a + (own_b - own_a) * alpha

  -- Optional: DoF
  if kf_a.dof_distance ~= nil then
    local dof_a = kf_a.dof_distance or 0
    local dof_b = kf_b.dof_distance or dof_a
    cam.dofDistance = dof_a + (dof_b - dof_a) * alpha
    local doff_a = kf_a.dof_factor or 0
    local doff_b = kf_b.dof_factor or doff_a
    cam.dofFactor = doff_a + (doff_b - doff_a) * alpha
  end
end

-- ============================================================================
-- Playback control
-- ============================================================================

local function startPlayback()
  if state.active then return end
  state.error = nil

  -- Load timeline if not loaded
  if state.timeline == nil then
    if not loadTimeline() then return end
  end

  -- Grab camera
  if ac.grabCamera == nil then
    state.error = 'ac.grabCamera not available'
    return
  end

  local ok, cam = pcall(function()
    return ac.grabCamera('Timeline Driver Spike')
  end)
  if not ok or cam == nil then
    state.error = 'Failed to grab camera: ' .. tostring(cam)
    return
  end

  state.cam = cam
  state.active = true
  state.current_shot_idx = -1
  state.total_frames = 0
  state.status = 'playing'
  ac.log('[Timeline Driver] Playback started')
end

local function stopPlayback()
  if state.cam then
    pcall(function() state.cam:dispose() end)
    state.cam = nil
  end
  state.active = false
  state.status = 'stopped'
  ac.log('[Timeline Driver] Playback stopped')
end

local function reloadTimeline()
  state.timeline = nil
  state.shots = nil
  state.error = nil
  if loadTimeline() then
    state.status = 'reloaded'
  end
end

-- ============================================================================
-- Per-frame update
-- ============================================================================

local function updateCamera()
  if not state.active then return end

  -- Read replay time via frame counter (proven in aidirector_spike)
  local sim = ac.getSim()
  if sim == nil then return end

  local frame = sim.replayCurrentFrame or 0
  local t = frame / state.sample_rate
  state.last_t = t

  -- Find active shot
  local shot, idx = findShot(t)

  if shot == nil then
    -- Outside timeline range — release camera
    ac.log(string.format('[Timeline Driver] t=%.3f outside timeline, releasing camera', t))
    stopPlayback()
    return
  end

  state.current_shot_idx = idx

  -- Find and interpolate keyframes
  local kf_a, kf_b, alpha = findKeyframes(shot, t)
  if kf_a == nil then return end

  -- Apply to camera
  local ok, err = pcall(applyKeyframe, kf_a, kf_b, alpha)
  if not ok then
    ac.log('[Timeline Driver] Apply error: ' .. tostring(err))
  end

  state.total_frames = state.total_frames + 1
end

-- ============================================================================
-- UI
-- ============================================================================

local function drawWindow()
  ui.text('=== TIMELINE DRIVER SPIKE ===')
  ui.text('CSP: ' .. state.csp)

  if state.error then
    ui.textWrapped('ERROR: ' .. state.error)
    ui.separator()
  end

  -- Timeline info
  if state.timeline then
    ui.text(string.format('Timeline: %d shots, %.1fs',
      #state.shots,
      state.timeline.summary and state.timeline.summary.duration_s or 0))
    ui.text(string.format('Sample rate: %.4f Hz', state.sample_rate))
  else
    ui.text('Timeline: not loaded')
  end

  -- Replay info
  local sim = ac.getSim()
  if sim then
    local frame = sim.replayCurrentFrame or 0
    local total = sim.replayFrames or 0
    local rate = sim.replayPlaybackRate or 0
    ui.text(string.format('Replay: frame %d/%d rate=%.1fx', frame, total, rate))
    ui.text(string.format('Time: %.3fs (frame / sample_rate)', frame / state.sample_rate))
  end

  ui.separator()

  -- Playback status
  ui.text('Status: ' .. state.status)
  if state.active then
    ui.text(string.format('t = %.3fs', state.last_t))
    if state.current_shot_idx > 0 and state.shots then
      local shot = state.shots[state.current_shot_idx]
      if shot then
        ui.text(string.format('Shot %d/%d: %s [%.1fs - %.1fs]',
          state.current_shot_idx, #state.shots,
          shot.rig or '?',
          shot.start_s, shot.end_s))
      end
    end
    ui.text(string.format('Frames driven: %d', state.total_frames))
  end

  ui.separator()

  -- Controls
  if not state.active then
    if ui.button('Play', vec2(100, 30)) then
      startPlayback()
    end
    ui.sameLine()
    if ui.button('Reload Timeline', vec2(150, 30)) then
      reloadTimeline()
    end
  else
    if ui.button('Stop', vec2(100, 30)) then
      stopPlayback()
    end
  end

  ui.separator()
  ui.textWrapped('File: ' .. state.timeline_file)
end

-- ============================================================================
-- Entry point
-- ============================================================================

function script.windowMain()
  if not state.probed then
    state.probed = true
    local ok, v = pcall(function() return ac.getPatchVersion() end)
    state.csp = ok and tostring(v) or 'unknown'
  end

  -- Per-frame camera update
  updateCamera()

  -- Draw UI
  drawWindow()
end
