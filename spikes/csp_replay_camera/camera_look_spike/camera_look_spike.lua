-- Camera Look Spike — CSP 0.3.0-preview595
--
-- Isolated diagnostic to experimentally determine:
--   1. Is transform.look a DIRECTION vector or a world-space TARGET point?
--   2. Does per-field assignment (.position, .look, .up, .side) mutate the
--      active camera, or is a full transform assignment required?
--   3. Does the CSP auto-reorthogonalise when only .look is changed?
--   4. What is the correct orientation for a chase camera?
--   5. Does per-frame camera update work without stuttering?
--
-- Design:
--   Six sequential phases, each held for a fixed duration.
--   Every phase logs: car.position, car.look, cam.transform.position,
--   cam.transform.look, cam.transform.up, cam.transform.side BEFORE and
--   AFTER the assignment.
--   Each phase has an explicit EXPECTED outcome so the visual result can be
--   compared to a prediction rather than interpreted in isolation.
--
-- This file does NOT modify any existing spike or the main project.

-- ============================================================================
-- Configuration
-- ============================================================================

local PHASE_HOLD = 5.0  -- seconds to hold each test phase

-- ============================================================================
-- State
-- ============================================================================

local state = {
  csp = 'unknown',
  probed = false,
  cam = nil,           -- GrabbedCamera or nil
  phase = 0,           -- 0=idle, 1..6=test phases, 7=done
  phase_start = 0,     -- os.clock() when current phase started
  frame_count = 0,     -- frames in current phase
  log = {},            -- { string, ... } for UI display
  error = nil,
  -- Snapshots from phase start for static tests
  anchor_car_pos = nil,
  anchor_car_look = nil,
  anchor_cam_pos = nil,
  -- Per-frame chase tracking
  chase_frames = 0,
  chase_readbacks = {},
}

-- ============================================================================
-- Utilities
-- ============================================================================

local function v3str(v)
  if v == nil then return 'nil' end
  local ok, s = pcall(function()
    return string.format('(%.4f, %.4f, %.4f)', v.x, v.y, v.z)
  end)
  return ok and s or tostring(v)
end

local function v3len(v)
  return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
end

local function v3norm(v)
  local len = v3len(v)
  if len < 1e-12 then return vec3(0, 0, 1) end
  return vec3(v.x / len, v.y / len, v.z / len)
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

local function logLine(text)
  state.log[#state.log + 1] = text
  ac.log('[Camera Look Spike] ' .. text)
end

--- Read the focused car's state. Returns (position, look) or (nil, nil).
local function readCar()
  local ok, car = pcall(function() return ac.getCar(0) end)
  if not ok or car == nil then return nil, nil end
  local pos = car.position
  local look = car.look
  if pos == nil then return nil, nil end
  if look == nil then
    -- fallback: try car.side to derive look
    local sOk, side = pcall(function() return car.side end)
    if sOk and side ~= nil then
      look = v3cross(vec3(0, 1, 0), vec3(side.x, side.y, side.z))
      look = v3norm(look)
    else
      look = vec3(0, 0, 1)
    end
  end
  return vec3(pos.x, pos.y, pos.z), vec3(look.x, look.y, look.z)
end

--- Log the full camera transform state.
local function logCamState(label)
  if state.cam == nil then
    logLine(label .. ': cam=nil')
    return
  end
  local t = state.cam.transform
  logLine(label .. ':')
  logLine('  .position = ' .. v3str(t.position) .. '  |len|=' .. string.format('%.2f', v3len(t.position)))
  logLine('  .look     = ' .. v3str(t.look) .. '  |len|=' .. string.format('%.6f', v3len(t.look)))
  logLine('  .up       = ' .. v3str(t.up) .. '  |len|=' .. string.format('%.6f', v3len(t.up)))
  logLine('  .side     = ' .. v3str(t.side) .. '  |len|=' .. string.format('%.6f', v3len(t.side)))
  logLine('  .fov      = ' .. string.format('%.2f', state.cam.fov))
end

--- Log the car state.
local function logCarState(carPos, carLook, label)
  logLine(label .. ':')
  logLine('  car.position = ' .. v3str(carPos))
  logLine('  car.look     = ' .. v3str(carLook) .. '  |len|=' .. string.format('%.6f', v3len(carLook)))
end

-- ============================================================================
-- Phase definitions
-- ============================================================================

local phases = {}

-- ---------- Phase 1: Readback test ----------
phases[1] = {
  name = 'READBACK TEST',
  description = 'Assigns position and look individually, then reads back.\n'
    .. 'Tests whether per-field assignment modifies the active camera.',
  expected = 'Camera should be 10m above car. Look readback should match\n'
    .. 'what was written. If readback differs, per-field assignment is broken.',
  setup = function(carPos, carLook)
    local abovePos = vec3(carPos.x, carPos.y + 10.0, carPos.z)
    local downVec = vec3(0, -1, 0)

    logLine('--- PHASE 1: READBACK TEST ---')
    logLine('Writing .position = ' .. v3str(abovePos))
    logLine('Writing .look = ' .. v3str(downVec))

    logCamState('BEFORE assignment')

    -- Write position
    state.cam.transform.position = abovePos
    -- Write look to straight down
    state.cam.transform.look = downVec
    state.cam.fov = 60

    logCamState('AFTER assignment')

    -- Check if readback matches
    local rb_pos = state.cam.transform.position
    local rb_look = state.cam.transform.look
    local pos_match = math.abs(rb_pos.x - abovePos.x) < 0.1
      and math.abs(rb_pos.y - abovePos.y) < 0.1
      and math.abs(rb_pos.z - abovePos.z) < 0.1
    local look_match = math.abs(rb_look.x - downVec.x) < 0.01
      and math.abs(rb_look.y - downVec.y) < 0.01
      and math.abs(rb_look.z - downVec.z) < 0.01

    logLine('POSITION readback matches written: ' .. tostring(pos_match))
    logLine('LOOK readback matches written: ' .. tostring(look_match))

    if not look_match then
      logLine('!! .look was normalised or modified by CSP')
      logLine('   written:  ' .. v3str(downVec))
      logLine('   readback: ' .. v3str(rb_look))
    end
  end,
  update = function(dt) end,
}

-- ---------- Phase 2: look = DIRECTION (unit vector toward car) ----------
phases[2] = {
  name = 'LOOK = DIRECTION (unit vec toward car)',
  description = 'Camera 8m behind, 3m above car.\n'
    .. 'Sets look = normalize(car_pos - cam_pos).\n'
    .. 'If look is a direction: car should be visible in centre.\n'
    .. 'If look is a target: camera points at a point near world origin.',
  expected = 'IF direction: correct chase view with car in frame.\n'
    .. 'IF target: completely wrong view (looking at world origin).',
  setup = function(carPos, carLook)
    local camPos = vec3(
      carPos.x - carLook.x * 8.0,
      carPos.y + 3.0,
      carPos.z - carLook.z * 8.0
    )
    local dirToCar = v3norm(vec3(
      carPos.x - camPos.x,
      carPos.y - camPos.y,
      carPos.z - camPos.z
    ))

    logLine('--- PHASE 2: LOOK = DIRECTION ---')
    logLine('cam_pos = ' .. v3str(camPos))
    logLine('direction to car = ' .. v3str(dirToCar) .. '  |len|=' .. string.format('%.6f', v3len(dirToCar)))

    state.cam.transform.position = camPos
    state.cam.transform.look = dirToCar
    state.cam.fov = 60

    logCamState('AFTER P2 assignment')
    state.anchor_cam_pos = camPos
  end,
  update = function(dt) end,
}

-- ---------- Phase 3: look = TARGET (world-space car position) ----------
phases[3] = {
  name = 'LOOK = TARGET (car world position)',
  description = 'Same camera position as Phase 2.\n'
    .. 'Sets look = car.position (raw world coordinates).\n'
    .. 'If look is a target: car should be visible in centre.\n'
    .. 'If look is a direction: camera points in a very specific world-scale direction.',
  expected = 'IF target: correct chase view with car in frame.\n'
    .. 'IF direction: completely wrong view (look vector has huge magnitude).',
  setup = function(carPos, carLook)
    local camPos = vec3(
      carPos.x - carLook.x * 8.0,
      carPos.y + 3.0,
      carPos.z - carLook.z * 8.0
    )

    logLine('--- PHASE 3: LOOK = TARGET ---')
    logLine('cam_pos = ' .. v3str(camPos))
    logLine('target (car_pos) = ' .. v3str(carPos) .. '  |len|=' .. string.format('%.2f', v3len(carPos)))

    state.cam.transform.position = camPos
    state.cam.transform.look = carPos
    state.cam.fov = 60

    logCamState('AFTER P3 assignment')

    -- Critical diagnostic: did CSP normalise it?
    local rb_look = state.cam.transform.look
    local rb_len = v3len(rb_look)
    logLine('READBACK .look |len| = ' .. string.format('%.6f', rb_len))
    if math.abs(rb_len - 1.0) < 0.01 then
      logLine('!! CSP normalised the value -> look is stored as direction')
    else
      logLine('CSP kept the raw value -> look may be a target')
    end
  end,
  update = function(dt) end,
}

-- ---------- Phase 4: Full basis (direction toward car + orthonormal) ----------
phases[4] = {
  name = 'FULL BASIS (direction + side + up)',
  description = 'Camera 8m behind car. Computes a full orthonormal basis\n'
    .. 'with look = direction toward car. Sets .side, .up, .look together.\n'
    .. 'Tests whether setting all three together produces correct orientation.',
  expected = 'Chase view with car in frame, correct up/horizon.\n'
    .. 'If not: the basis convention (handedness or axis mapping) is wrong.',
  setup = function(carPos, carLook)
    local camPos = vec3(
      carPos.x - carLook.x * 8.0,
      carPos.y + 3.0,
      carPos.z - carLook.z * 8.0
    )
    local fwd = v3norm(vec3(
      carPos.x - camPos.x,
      carPos.y - camPos.y,
      carPos.z - camPos.z
    ))
    local worldUp = vec3(0, 1, 0)
    local right = v3norm(v3cross(fwd, worldUp))
    local up = v3cross(right, fwd)

    logLine('--- PHASE 4: FULL BASIS (direction) ---')
    logLine('cam_pos = ' .. v3str(camPos))
    logLine('forward = ' .. v3str(fwd))
    logLine('right   = ' .. v3str(right))
    logLine('up      = ' .. v3str(up))
    logLine('dot(fwd, right) = ' .. string.format('%.6f', v3dot(fwd, right)))
    logLine('dot(fwd, up)    = ' .. string.format('%.6f', v3dot(fwd, up)))
    logLine('dot(right, up)  = ' .. string.format('%.6f', v3dot(right, up)))

    state.cam.transform.position = camPos
    state.cam.transform.look = fwd
    state.cam.transform.side = right
    state.cam.transform.up = up
    state.cam.fov = 60

    logCamState('AFTER P4 assignment')
  end,
  update = function(dt) end,
}

-- ---------- Phase 5: Negated look (OpenGL convention: cam looks down -Z) ----------
phases[5] = {
  name = 'NEGATED LOOK (-forward + side + up)',
  description = 'Same as Phase 4 but look = -forward (OpenGL camera convention).\n'
    .. 'In OpenGL, the camera looks down its -Z axis, so the stored .look\n'
    .. 'vector might be the OPPOSITE of the viewing direction.',
  expected = 'If CSP uses -Z viewing: this should show correct chase view.\n'
    .. 'If Phase 4 was already correct: this shows the reverse (sky/behind).',
  setup = function(carPos, carLook)
    local camPos = vec3(
      carPos.x - carLook.x * 8.0,
      carPos.y + 3.0,
      carPos.z - carLook.z * 8.0
    )
    local fwd = v3norm(vec3(
      carPos.x - camPos.x,
      carPos.y - camPos.y,
      carPos.z - camPos.z
    ))
    local negFwd = vec3(-fwd.x, -fwd.y, -fwd.z)
    local worldUp = vec3(0, 1, 0)
    -- Build basis around negated look
    local right = v3norm(v3cross(negFwd, worldUp))
    local up = v3cross(right, negFwd)

    logLine('--- PHASE 5: NEGATED LOOK (-forward) ---')
    logLine('cam_pos   = ' .. v3str(camPos))
    logLine('forward   = ' .. v3str(fwd))
    logLine('neg_fwd   = ' .. v3str(negFwd))
    logLine('right     = ' .. v3str(right))
    logLine('up        = ' .. v3str(up))

    state.cam.transform.position = camPos
    state.cam.transform.look = negFwd
    state.cam.transform.side = right
    state.cam.transform.up = up
    state.cam.fov = 60

    logCamState('AFTER P5 assignment')
  end,
  update = function(dt) end,
}

-- ---------- Phase 6: Per-frame chase (tests both conventions) ----------
phases[6] = {
  name = 'PER-FRAME CHASE (5 seconds)',
  description = 'Every frame: reads car position, places camera behind,\n'
    .. 'sets look toward car. Uses the direction convention.\n'
    .. 'Logs readback every 30 frames to verify continuous update.',
  expected = 'Camera smoothly follows the car during replay playback.\n'
    .. 'If the car is visible and the camera follows: per-frame update works.\n'
    .. 'If it stutters or drifts: assignment may not persist across frames.',
  setup = function(carPos, carLook)
    logLine('--- PHASE 6: PER-FRAME CHASE ---')
    logLine('Starting continuous per-frame update...')
    state.chase_frames = 0
    state.chase_readbacks = {}
    state.cam.fov = 55
  end,
  update = function(dt)
    local carPos, carLook = readCar()
    if carPos == nil then return end

    -- Place camera behind car along its heading
    local camPos = vec3(
      carPos.x - carLook.x * 8.0,
      carPos.y + 3.0,
      carPos.z - carLook.z * 8.0
    )
    local fwd = v3norm(vec3(
      carPos.x - camPos.x,
      carPos.y - camPos.y,
      carPos.z - camPos.z
    ))
    local worldUp = vec3(0, 1, 0)
    local right = v3norm(v3cross(fwd, worldUp))
    local up = v3cross(right, fwd)

    state.cam.transform.position = camPos
    state.cam.transform.look = fwd
    state.cam.transform.side = right
    state.cam.transform.up = up

    state.chase_frames = state.chase_frames + 1

    -- Log every 30 frames
    if state.chase_frames % 30 == 0 then
      local rb = {
        frame = state.chase_frames,
        car_pos = v3str(carPos),
        cam_pos = v3str(state.cam.transform.position),
        cam_look = v3str(state.cam.transform.look),
        look_len = string.format('%.6f', v3len(state.cam.transform.look)),
      }
      state.chase_readbacks[#state.chase_readbacks + 1] = rb
      logLine(string.format('  chase frame %d: car=%s cam=%s look=%s |look|=%s',
        rb.frame, rb.car_pos, rb.cam_pos, rb.cam_look, rb.look_len))
    end
  end,
}

-- ============================================================================
-- Phase management
-- ============================================================================

local function startPhase(n)
  state.phase = n
  state.phase_start = os.clock()
  state.frame_count = 0

  local carPos, carLook = readCar()
  if carPos == nil then
    state.error = 'Cannot read car state. Is a replay playing?'
    state.phase = 0
    return
  end

  state.anchor_car_pos = carPos
  state.anchor_car_look = carLook

  logLine('')
  logLine('====================================')
  logLine('PHASE ' .. n .. ': ' .. phases[n].name)
  logLine('====================================')
  logCarState(carPos, carLook, 'Car state at phase start')

  local ok, err = pcall(phases[n].setup, carPos, carLook)
  if not ok then
    logLine('!! SETUP ERROR: ' .. tostring(err))
  end
end

local function advancePhase()
  local elapsed = os.clock() - state.phase_start
  if elapsed < PHASE_HOLD then return end

  if state.phase >= #phases then
    -- All done
    logLine('')
    logLine('====================================')
    logLine('ALL PHASES COMPLETE')
    logLine('====================================')

    -- Summary of per-frame chase
    if #state.chase_readbacks > 0 then
      logLine('Per-frame chase: ' .. state.chase_frames .. ' frames total')
      logLine('Readback samples: ' .. #state.chase_readbacks)
    end

    -- Release camera
    if state.cam then
      pcall(function() state.cam:dispose() end)
      state.cam = nil
    end
    state.phase = 7  -- done
    return
  end

  startPhase(state.phase + 1)
end

-- ============================================================================
-- Control
-- ============================================================================

local function startAllTests()
  if state.phase ~= 0 and state.phase ~= 7 then return end

  state.log = {}
  state.error = nil

  -- CSP version
  local ok, v = pcall(function() return ac.getPatchVersion() end)
  state.csp = ok and tostring(v) or 'unknown'
  logLine('CSP version: ' .. state.csp)

  -- Grab camera
  if ac.grabCamera == nil then
    state.error = 'ac.grabCamera not available'
    return
  end

  local gOk, cam = pcall(function()
    return ac.grabCamera('Camera Look Spike')
  end)
  if not gOk or cam == nil then
    state.error = 'Failed to grab camera: ' .. tostring(cam)
    return
  end
  state.cam = cam

  logLine('Camera grabbed successfully')
  logCamState('Initial camera state')

  startPhase(1)
end

local function stopTests()
  if state.cam then
    pcall(function() state.cam:dispose() end)
    state.cam = nil
  end
  state.phase = 7
  logLine('Tests stopped manually')
end

-- ============================================================================
-- UI
-- ============================================================================

local scrollPos = 0

local function drawWindow()
  ui.text('=== CAMERA LOOK SPIKE ===')
  ui.text('CSP: ' .. state.csp)
  ui.text('')

  if state.error then
    ui.textWrapped('ERROR: ' .. state.error)
    ui.text('')
  end

  -- Controls
  if state.phase == 0 or state.phase == 7 then
    ui.textWrapped(
      'This spike runs 6 sequential tests (' .. PHASE_HOLD .. 's each) '
      .. 'to determine how GrabbedCamera.transform.look works. '
      .. 'Watch the viewport during each phase and note what you see.'
    )
    ui.text('')
    if ui.button('Start All Tests', vec2(200, 30)) then
      startAllTests()
    end
  else
    local p = phases[state.phase]
    local elapsed = os.clock() - state.phase_start
    local remaining = math.max(0, PHASE_HOLD - elapsed)
    ui.text(string.format('PHASE %d/%d: %s', state.phase, #phases, p.name))
    ui.text(string.format('Next phase in: %.1fs', remaining))
    ui.separator()
    ui.textWrapped('EXPECTED: ' .. p.expected)
    ui.text('')
    if ui.button('Skip Phase', vec2(120, 25)) then
      state.phase_start = 0
    end
    ui.sameLine()
    if ui.button('Stop All', vec2(120, 25)) then
      stopTests()
    end
  end

  -- Log display
  ui.separator()
  ui.text('=== LOG ===')

  -- Show last ~25 lines
  local startLine = math.max(1, #state.log - 25)
  for i = startLine, #state.log do
    ui.text(state.log[i])
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
  end

  -- Run phase update logic
  if state.phase >= 1 and state.phase <= #phases then
    state.frame_count = state.frame_count + 1
    -- Call per-frame update for current phase
    local ok, err = pcall(phases[state.phase].update, 1 / 60)
    if not ok then
      logLine('!! UPDATE ERROR: ' .. tostring(err))
    end
    advancePhase()
  end

  drawWindow()
end
