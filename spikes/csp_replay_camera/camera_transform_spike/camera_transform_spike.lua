-- Camera Transform Spike
--
-- Minimal test to establish the exact mat4x4 convention CSP uses for
-- ac.GrabbedCamera.transform. This is the last unknown before a real
-- camera driver can be built.
--
-- The test:
--   1. Reads the player car's current position as an anchor point.
--   2. Grabs the camera via ac.grabCamera.
--   3. Cycles through a series of known positions and look directions,
--      holding each for a few seconds.
--   4. Records what worked and what the visual result was.
--   5. Releases the camera.
--
-- Each test pose has a DESCRIPTION of what you should see. Compare what
-- you actually see with the description to determine the mat4x4 convention.
--
-- The tests are:
--   A) Position only: place camera 5m above the car, keep default orientation
--   B) Look forward along car's heading (Z axis)
--   C) Look to the right (X axis)
--   D) Look straight down (-Y axis)
--   E) Look at 45 degrees down-forward
--   F) Dutch tilt: look forward with 30-degree roll
--
-- After all tests, a summary is printed in the window and logged.

local HOLD_SECONDS = 4.0  -- how long to hold each test pose
local HEIGHT_ABOVE = 5.0  -- metres above car for test positions
local DISTANCE_BEHIND = 8.0  -- metres behind car for chase view tests

local state = {
  probed = false,
  csp = 'unknown',
  cam = nil,             -- GrabbedCamera reference
  active = false,        -- currently running the test sequence
  test_index = 0,        -- current test (0 = not started)
  test_start_time = 0,   -- when current test started (os.clock)
  car_pos = nil,         -- car world position at start
  car_look = nil,        -- car forward direction at start
  results = {},          -- { name, description, expected, observed, status }
  error = nil,
  phase = 'idle',        -- 'idle', 'running', 'done'
}

-- ============================================================================
-- Test pose definitions
-- ============================================================================

-- Build a mat4x4 from position + look + up.
-- This is the convention we're TESTING. We try the most common OpenGL-style
-- convention first:
--   right   = normalize(cross(look, up))
--   true_up = cross(right, look)        -- re-orthogonalise
--   mat4x4:
--     row 0 = right
--     row 1 = true_up
--     row 2 = -look (camera looks down -Z in OpenGL convention)
--     row 3 = position
--
-- If the visual results are wrong, we know to try alternatives (transposed,
-- +Z look, row/column swap, etc).

local function buildTransform_OpenGL(pos, look, up)
  local l = look:normalize()
  local r = math.cross(l, up):normalize()
  local u = math.cross(r, l):normalize()

  -- mat4x4 in CSP: we try setting .row0/.row1/.row2/.row3
  -- or .side/.up/.look/.position if those properties exist.
  -- The spike reports which path worked.
  return {
    right = r,
    up = u,
    negLook = vec3(-l.x, -l.y, -l.z),
    look = l,
    position = pos,
  }
end

-- ============================================================================
-- Test sequence
-- ============================================================================

local tests = {}  -- populated in startTests after we know car position

local function defineTests(carPos, carLook)
  local carRight = math.cross(carLook, vec3(0, 1, 0)):normalize()
  local abovePos = vec3(carPos.x, carPos.y + HEIGHT_ABOVE, carPos.z)
  local behindPos = vec3(
    carPos.x - carLook.x * DISTANCE_BEHIND,
    carPos.y + HEIGHT_ABOVE,
    carPos.z - carLook.z * DISTANCE_BEHIND
  )

  tests = {
    {
      name = 'A: Position only',
      description = 'Camera 5m above car. Default orientation (no transform change).',
      expected = 'Camera should be above the car, looking in whatever the default direction is.',
      apply = function(cam)
        cam.transform.position = abovePos
        -- Don't touch orientation: we want to see what the default is
        cam.fov = 60
      end,
    },
    {
      name = 'B: Look forward (car heading)',
      description = 'Camera 8m behind and 5m above car, looking forward along car heading.',
      expected = 'Chase cam view: car in lower centre, road ahead visible.',
      apply = function(cam)
        local t = buildTransform_OpenGL(behindPos, carLook, vec3(0, 1, 0))
        cam.transform.position = behindPos
        cam.transform.side = t.right
        cam.transform.up = t.up
        cam.transform.look = t.negLook
        cam.fov = 60
      end,
    },
    {
      name = 'B2: Look forward (negated Z)',
      description = 'Same as B but with look = +forward instead of -forward.',
      expected = 'If B was wrong (looking backward), this should be correct chase view.',
      apply = function(cam)
        local t = buildTransform_OpenGL(behindPos, carLook, vec3(0, 1, 0))
        cam.transform.position = behindPos
        cam.transform.side = t.right
        cam.transform.up = t.up
        cam.transform.look = t.look
        cam.fov = 60
      end,
    },
    {
      name = 'C: Look right',
      description = 'Camera above car, looking to the right of the car.',
      expected = 'You should see the landscape to the right side of the track.',
      apply = function(cam)
        local t = buildTransform_OpenGL(abovePos, carRight, vec3(0, 1, 0))
        cam.transform.position = abovePos
        cam.transform.side = t.right
        cam.transform.up = t.up
        cam.transform.look = t.negLook
        cam.fov = 60
      end,
    },
    {
      name = 'D: Look straight down',
      description = 'Camera 10m above car, looking straight down.',
      expected = 'Bird\'s eye view: car directly below, road from above.',
      apply = function(cam)
        local highPos = vec3(carPos.x, carPos.y + 10.0, carPos.z)
        local t = buildTransform_OpenGL(highPos, vec3(0, -1, 0), carLook)
        cam.transform.position = highPos
        cam.transform.side = t.right
        cam.transform.up = t.up
        cam.transform.look = t.negLook
        cam.fov = 60
      end,
    },
    {
      name = 'E: 45° down-forward',
      description = 'Camera behind and above, looking 45 degrees downward toward the car.',
      expected = 'Angled chase: car visible in lower half, road surface visible.',
      apply = function(cam)
        local lookDir = vec3(carLook.x, -1, carLook.z):normalize()
        local highBehind = vec3(
          carPos.x - carLook.x * 12.0,
          carPos.y + 8.0,
          carPos.z - carLook.z * 12.0
        )
        local t = buildTransform_OpenGL(highBehind, lookDir, vec3(0, 1, 0))
        cam.transform.position = highBehind
        cam.transform.side = t.right
        cam.transform.up = t.up
        cam.transform.look = t.negLook
        cam.fov = 50
      end,
    },
    {
      name = 'F: FOV test (wide)',
      description = 'Same position as E but with very wide FOV (100°).',
      expected = 'Same view as E but much wider, fish-eye distortion.',
      apply = function(cam)
        local lookDir = vec3(carLook.x, -1, carLook.z):normalize()
        local highBehind = vec3(
          carPos.x - carLook.x * 12.0,
          carPos.y + 8.0,
          carPos.z - carLook.z * 12.0
        )
        local t = buildTransform_OpenGL(highBehind, lookDir, vec3(0, 1, 0))
        cam.transform.position = highBehind
        cam.transform.side = t.right
        cam.transform.up = t.up
        cam.transform.look = t.negLook
        cam.fov = 100
      end,
    },
    {
      name = 'G: FOV test (telephoto)',
      description = 'Same position as E but with narrow telephoto FOV (15°).',
      expected = 'Same view as E but very zoomed in, compressed perspective.',
      apply = function(cam)
        local lookDir = vec3(carLook.x, -1, carLook.z):normalize()
        local highBehind = vec3(
          carPos.x - carLook.x * 12.0,
          carPos.y + 8.0,
          carPos.z - carLook.z * 12.0
        )
        local t = buildTransform_OpenGL(highBehind, lookDir, vec3(0, 1, 0))
        cam.transform.position = highBehind
        cam.transform.side = t.right
        cam.transform.up = t.up
        cam.transform.look = t.negLook
        cam.fov = 15
      end,
    },
  }
end

-- ============================================================================
-- Execution
-- ============================================================================

local function getCarState()
  local ok, car = pcall(function() return ac.getCar(0) end)
  if not ok or car == nil then return nil, nil end

  local pos = car.position
  if pos == nil then return nil, nil end

  -- Try to get heading from the car's transform or velocity
  local look = nil

  -- Method 1: car.look (forward direction)
  local lookOk, carLook = pcall(function() return car.look end)
  if lookOk and carLook ~= nil then
    look = vec3(carLook.x, carLook.y, carLook.z)
  end

  -- Method 2: from side vector
  if look == nil then
    local sideOk, carSide = pcall(function() return car.side end)
    if sideOk and carSide ~= nil then
      look = math.cross(vec3(0, 1, 0), vec3(carSide.x, carSide.y, carSide.z)):normalize()
    end
  end

  -- Method 3: fallback to Z+ direction
  if look == nil then
    look = vec3(0, 0, 1)
  end

  return vec3(pos.x, pos.y, pos.z), look
end

local function grabCamera()
  if ac.grabCamera == nil then
    state.error = 'ac.grabCamera is not available in this CSP version'
    return false
  end

  local ok, cam = pcall(function()
    return ac.grabCamera('Camera Transform Spike')
  end)

  if not ok or cam == nil then
    state.error = 'Failed to grab camera: ' .. tostring(cam)
    return false
  end

  state.cam = cam
  return true
end

local function releaseCamera()
  if state.cam then
    pcall(function() state.cam:dispose() end)
    state.cam = nil
  end
end

local function startTests()
  if state.active then return end

  -- Get car position
  local pos, look = getCarState()
  if pos == nil then
    state.error = 'Cannot read car position. Make sure a replay is playing.'
    return
  end

  state.car_pos = pos
  state.car_look = look
  state.results = {}
  state.error = nil

  -- Define the test poses based on actual car position
  defineTests(pos, look)

  -- Grab camera
  if not grabCamera() then return end

  -- Log the car state for the record
  ac.log('=== CAMERA TRANSFORM SPIKE: START ===')
  ac.log(string.format('car_pos: (%.2f, %.2f, %.2f)', pos.x, pos.y, pos.z))
  ac.log(string.format('car_look: (%.3f, %.3f, %.3f)', look.x, look.y, look.z))

  -- Inspect GrabbedCamera properties
  ac.log('--- GrabbedCamera fields ---')
  local fields = { 'transform', 'fov', 'dofDistance', 'dofFactor', 'exposure', 'ownShare' }
  for _, f in ipairs(fields) do
    local fOk, fVal = pcall(function() return state.cam[f] end)
    ac.log(string.format('  %s: %s (%s)', f, tostring(fVal), type(fVal)))
  end

  -- Inspect transform sub-fields
  local tOk, transform = pcall(function() return state.cam.transform end)
  if tOk and transform ~= nil then
    ac.log('--- transform fields ---')
    local tfields = { 'position', 'look', 'up', 'side', 'row0', 'row1', 'row2', 'row3' }
    for _, f in ipairs(tfields) do
      local tfOk, tfVal = pcall(function() return transform[f] end)
      if tfOk and tfVal ~= nil then
        local vOk, vStr = pcall(function()
          if tfVal.z ~= nil then
            return string.format('(%.3f, %.3f, %.3f)', tfVal.x, tfVal.y, tfVal.z)
          end
          return tostring(tfVal)
        end)
        ac.log(string.format('  .%s: %s [%s]', f, vOk and vStr or tostring(tfVal), type(tfVal)))
      else
        ac.log(string.format('  .%s: NOT AVAILABLE', f))
      end
    end
  end

  state.active = true
  state.test_index = 1
  state.test_start_time = os.clock()
  state.phase = 'running'

  -- Apply first test
  local applyOk, applyErr = pcall(tests[1].apply, state.cam)
  state.results[1] = {
    name = tests[1].name,
    description = tests[1].description,
    expected = tests[1].expected,
    applied = applyOk,
    apply_error = not applyOk and tostring(applyErr) or nil,
    observed = '',  -- user fills this in mentally
  }
  if applyOk then
    ac.log(string.format('TEST %d: %s — applied OK', 1, tests[1].name))
  else
    ac.log(string.format('TEST %d: %s — APPLY FAILED: %s', 1, tests[1].name, tostring(applyErr)))
  end
end

local function advanceTest()
  if not state.active then return end

  local elapsed = os.clock() - state.test_start_time
  if elapsed < HOLD_SECONDS then return end

  -- Move to next test
  state.test_index = state.test_index + 1

  if state.test_index > #tests then
    -- All done
    releaseCamera()
    state.active = false
    state.phase = 'done'
    ac.log('=== CAMERA TRANSFORM SPIKE: COMPLETE ===')
    ac.log(string.format('Tests run: %d', #tests))
    for i, r in ipairs(state.results) do
      ac.log(string.format('  %d. %s — %s', i, r.name, r.applied and 'OK' or ('FAILED: ' .. (r.apply_error or '?'))))
    end
    return
  end

  -- Apply next test
  state.test_start_time = os.clock()
  local test = tests[state.test_index]
  local applyOk, applyErr = pcall(test.apply, state.cam)
  state.results[state.test_index] = {
    name = test.name,
    description = test.description,
    expected = test.expected,
    applied = applyOk,
    apply_error = not applyOk and tostring(applyErr) or nil,
    observed = '',
  }
  if applyOk then
    ac.log(string.format('TEST %d: %s — applied OK', state.test_index, test.name))
  else
    ac.log(string.format('TEST %d: %s — APPLY FAILED: %s', state.test_index, test.name, tostring(applyErr)))
  end
end

local function stopTests()
  releaseCamera()
  state.active = false
  state.phase = 'done'
end

-- ============================================================================
-- UI
-- ============================================================================

local function drawWindow()
  ui.text('=== CAMERA TRANSFORM SPIKE ===')
  ui.text('csp: ' .. state.csp)

  if state.car_pos then
    ui.text(string.format('car: (%.1f, %.1f, %.1f)', state.car_pos.x, state.car_pos.y, state.car_pos.z))
    ui.text(string.format('heading: (%.3f, %.3f, %.3f)', state.car_look.x, state.car_look.y, state.car_look.z))
  end

  if state.error then
    ui.separator()
    ui.textWrapped('ERROR: ' .. state.error)
  end

  ui.separator()

  if state.phase == 'idle' then
    ui.textWrapped('This spike grabs the camera and cycles through ' .. #tests .. ' test orientations, '
      .. HOLD_SECONDS .. 's each. Watch the viewport and note if the view matches the description.')
    ui.text('')
    if ui.button('Start Test Sequence') then
      startTests()
    end
  elseif state.phase == 'running' then
    local test = tests[state.test_index]
    local elapsed = os.clock() - state.test_start_time
    local remaining = math.max(0, HOLD_SECONDS - elapsed)

    ui.text(string.format('TEST %d/%d: %s', state.test_index, #tests, test.name))
    ui.text(string.format('Next in: %.1fs', remaining))
    ui.separator()
    ui.textWrapped('EXPECTED: ' .. test.expected)

    local r = state.results[state.test_index]
    if r and not r.applied then
      ui.textWrapped('!! APPLY FAILED: ' .. (r.apply_error or '?'))
    end

    ui.text('')
    if ui.button('Stop Early') then
      stopTests()
    end
    if ui.button('Skip to Next') then
      state.test_start_time = 0  -- force advance on next frame
    end
  elseif state.phase == 'done' then
    ui.text('=== ALL TESTS COMPLETE ===')
    ui.text(string.format('Tests: %d total', #state.results))
    ui.separator()

    for i, r in ipairs(state.results) do
      local status = r.applied and 'OK' or 'FAILED'
      ui.text(string.format('%d. [%s] %s', i, status, r.name))
      if not r.applied then
        ui.textWrapped('   Error: ' .. (r.apply_error or '?'))
      end
    end

    ui.separator()
    ui.textWrapped('Check the CSP log for detailed field inspection. '
      .. 'Report which tests showed the CORRECT view and which looked wrong.')

    if ui.button('Run Again') then
      state.phase = 'idle'
      state.results = {}
      state.error = nil
    end
  end

  -- Always show transform field inspection results
  ui.separator()
  ui.text('=== TRANSFORM FIELD INSPECTION ===')
  ui.textWrapped('(Check CSP log for full output of available .transform fields: '
    .. '.position, .look, .up, .side, .row0-.row3)')
end

-- ============================================================================
-- Entry point
-- ============================================================================

function script.windowMain()
  if not state.probed then
    state.probed = true
    local ok, version = pcall(function() return ac.getPatchVersion() end)
    state.csp = ok and tostring(version) or 'unknown'

    -- Pre-read car state for display even before tests start
    state.car_pos, state.car_look = getCarState()

    -- Define tests so we know the count for the idle message
    if state.car_pos and state.car_look then
      defineTests(state.car_pos, state.car_look)
    else
      tests = {}  -- no car, no tests to define yet
    end
  end

  if state.active then
    advanceTest()
  end

  drawWindow()
end
