-- Answers two questions and scans the track terrain. It is not the beginning of an exporter.
--
--   1. Does ac.grabCamera work while a replay is playing?
--   2. Can a script see where in the replay it is?
--   3. (NEW) Scan the track terrain: build a grid of ground heights and sight lines.
--
-- Everything here is defensive, because the whole point is to find out what
-- exists. A call that is not there must be reported as not there, not crash the
-- app and leave the question open -- so every probe goes through pcall and
-- reports what it found, including nothing.

local probed = false

local result = {
  csp = 'not established',
  replay = 'not established',
  grab = 'not established',
  clock = 'not established',
}

-- Scan state
local scanning = false
local scan_progress = 0
local scan_total_stations = 0
local scan_current_station = 0
local scan_grid = {}
local scan_result_file = ''
local scan_error = nil
local api_available = false
local api_diagnostics = {}  -- Detailed diagnostic info for each API

-- Raycast test results
local raycast_test = {
  enabled = false,
  ray_creation = nil,
  ray_type = nil,
  physics_method = nil,
  physics_call = nil,
  physics_result = nil,
  physics_error = nil,
  track_method = nil,
  track_call = nil,
  track_result = nil,
  track_error = nil,
}

--- Call something that may not exist, and say so when it does not.
-- @param label What is being probed, for the report.
-- @param fn The call.
-- @return The value, or nil.
local function probe(label, fn)
  local ok, value = pcall(fn)
  if not ok then
    return nil, string.format('%s raised: %s', label, tostring(value))
  end
  if value == nil then
    return nil, string.format('%s returned nil', label)
  end
  return value, nil
end

--- Report the patch version, so a negative result can be attributed.
local function checkVersion()
  local version = probe('ac.getPatchVersion', function() return ac.getPatchVersion() end)
  result.csp = version and tostring(version) or 'unknown'
end

--- Establish that this really is a replay. A live session answers nothing.
local function checkReplay()
  local sim = probe('ac.getSim', function() return ac.getSim() end)
  if sim == nil then
    result.replay = 'ac.getSim is unavailable'
    return false
  end
  if sim.isReplayActive == nil then
    result.replay = 'sim has no isReplayActive field'
    return false
  end
  if not sim.isReplayActive then
    result.replay = 'NOT a replay -- load a replay and reopen this app'
    return false
  end
  result.replay = 'yes, a replay is playing'
  return true
end

--- The first question: can the camera be taken during playback?
local function checkGrab()
  if ac.grabCamera == nil then
    result.grab = 'NO -- ac.grabCamera does not exist in this patch version'
    return
  end

  local grabbed, why = probe('ac.grabCamera', function()
    return ac.grabCamera('Assetto AI Director spike')
  end)
  if grabbed == nil then
    -- The grab is exclusive and returns the holder's name when refused. Saying
    -- which script has it is the difference between "this cannot work" and
    -- "close the other tool and try again".
    result.grab = 'REFUSED -- ' .. (why or 'no reason given; another script may hold the camera')
    return
  end

  local moved = pcall(function()
    -- One metre, for two seconds, then handed straight back. Enough to see, far
    -- too little to be mistaken for the app doing something useful.
    grabbed.transform.position:add(vec3(0, 1, 0))
    grabbed.fov = 45
  end)
  result.grab = moved
    and 'YES -- the camera was grabbed and moved during replay playback'
    or 'PARTIAL -- grabbed, but writing the transform failed'

  setTimeout(function()
    pcall(function() grabbed:dispose() end)
  end, 2)
end

--- The second question: is there a replay clock to key a timeline to?
local function checkClock()
  local sim = probe('ac.getSim', function() return ac.getSim() end)
  if sim == nil then
    result.clock = 'ac.getSim is unavailable'
    return
  end

  -- Every field that could plausibly carry playback position. Reported by name
  -- so a negative result says what was actually looked for, and a later patch
  -- version can be re-checked against the same list.
  local candidates = {
    'replayCurrentFrame',
    'replayFrames',
    'replayPlaybackRate',
    'timeSeconds',
    'timeTotalSeconds',
    'currentSessionTime',
  }
  local found = {}
  for _, name in ipairs(candidates) do
    if sim[name] ~= nil then
      found[#found + 1] = string.format('%s=%s', name, tostring(sim[name]))
    end
  end

  result.clock = #found > 0
    and ('FOUND -- ' .. table.concat(found, ', '))
    or ('NONE of: ' .. table.concat(candidates, ', '))
end

--- Run every probe once.
local function runProbes()
  checkVersion()
  local isReplay = checkReplay()
  checkGrab()
  checkClock()

  ac.log('=== RESULT ===')
  ac.log('csp     : ' .. result.csp)
  ac.log('replay  : ' .. result.replay)
  ac.log('grab    : ' .. result.grab)
  ac.log('clock   : ' .. result.clock)
  if not isReplay then
    ac.log('note    : the grab and clock results do not answer the question '
      .. 'unless a replay was playing.')
  end
end

-- ============================================================================
-- Terrain scanning logic
-- ============================================================================

local STATION_SPACING = 5.0  -- metres between stations along track
local PROFILE_SPACING = 2.0  -- metres between lateral profile points
local PROFILE_RANGE = 40.0   -- metres from -40 to +40
local RAY_LENGTH = 200.0     -- metres, long enough to hit anything

--- Detailed API diagnostic: distinguishes function existence from runtime availability.
-- States: FUNCTION_MISSING, FUNCTION_AVAILABLE, CALL_FAILED, RETURNED_NIL, AVAILABLE
local function diagnoseAPI(name, getFunc, callFunc)
  local diag = {
    name = name,
    exists = false,
    type = 'nil',
    state = 'FUNCTION_MISSING',
    callResult = nil,
    callError = nil,
    resultType = 'nil',
  }

  -- Step 1: Check if function exists
  local ok, func = pcall(getFunc)
  if not ok then
    diag.state = 'FUNCTION_MISSING'
    diag.callError = 'get failed: ' .. tostring(func)
    return diag
  end

  diag.exists = (func ~= nil)
  diag.type = type(func)

  if not diag.exists then
    diag.state = 'FUNCTION_MISSING'
    return diag
  end

  -- Step 2: Function exists, try to call it
  if callFunc then
    local callOk, callResult = pcall(callFunc, func)
    if not callOk then
      diag.state = 'CALL_FAILED'
      diag.callError = tostring(callResult)
      return diag
    end

    diag.callResult = callResult
    diag.resultType = type(callResult)

    if callResult == nil then
      diag.state = 'RETURNED_NIL'
    else
      diag.state = 'AVAILABLE'
    end
  else
    -- No call function provided, just check existence
    diag.state = 'FUNCTION_AVAILABLE'
  end

  return diag
end

--- Alternative API diagnostic: tests APIs that MAY replace missing ones.
-- Crucial difference from diagnoseAPI: a nil result does NOT mean the API is broken.
-- Some APIs return nil for specific inputs (e.g. world point far from track).
-- We only report FUNCTION_MISSING or CALL_FAILED as real failures.
-- @param name Display name
-- @param getFunc Returns the function object (or nil if absent)
-- @param testArgsStr Human-readable description of test arguments (for UI)
-- @param callFunc Calls the function with test arguments, returns the result
local function testAlternativeAPI(name, getFunc, testArgsStr, callFunc)
  local diag = {
    name = name,
    exists = false,
    type = 'nil',
    state = 'FUNCTION_MISSING',
    testArgs = testArgsStr,
    resultStr = '',
  }

  -- Step 1: existence check
  local ok, func = pcall(getFunc)
  if not ok then
    diag.state = 'FUNCTION_MISSING'
    diag.resultStr = 'get error: ' .. tostring(func)
    return diag
  end
  if func == nil then
    diag.state = 'FUNCTION_MISSING'
    diag.resultStr = 'function reference is nil'
    return diag
  end

  diag.exists = true
  diag.type = type(func)

  -- Step 2: call
  local callOk, callResult = pcall(callFunc, func)
  if not callOk then
    diag.state = 'CALL_FAILED'
    diag.resultStr = 'error: ' .. tostring(callResult)
    return diag
  end

  if callResult == nil then
    -- nil result is NOT a failure for these APIs: the test input may simply be
    -- invalid for the current track (e.g. world origin far from the track).
    diag.state = 'FUNCTION_AVAILABLE'
    diag.resultStr = 'returned nil (test input may be outside track)'
    return diag
  end

  diag.state = 'AVAILABLE'

  -- Format the result for display. vec3/vec2 in CSP have x/y/z fields.
  local fmtOk, fmtStr = pcall(function()
    if callResult.z ~= nil then
      return string.format('vec3(%.3f, %.3f, %.3f)', callResult.x, callResult.y, callResult.z)
    elseif callResult.y ~= nil then
      return string.format('vec2(%.3f, %.3f)', callResult.x, callResult.y)
    end
    local s = tostring(callResult)
    if s == '' then return '(empty string)' end
    return s
  end)
  if fmtOk then
    diag.resultStr = fmtStr
  else
    diag.resultStr = tostring(callResult)
  end

  return diag
end

--- Comprehensive API diagnostic: checks all terrain scan APIs with detailed status.
local function runFullAPIDiagnostic()
  api_diagnostics = {}

  -- ac.getTrackSpline
  api_diagnostics.getTrackSpline = diagnoseAPI(
    'ac.getTrackSpline',
    function() return ac.getTrackSpline end,
    function(func) return func() end
  )

  -- ac.getDocumentsPath
  api_diagnostics.getDocumentsPath = diagnoseAPI(
    'ac.getDocumentsPath',
    function() return ac.getDocumentsPath end,
    function(func) return func() end
  )

  -- ac.getLayoutName
  api_diagnostics.getLayoutName = diagnoseAPI(
    'ac.getLayoutName',
    function() return ac.getLayoutName end,
    function(func) return func() end
  )

  -- ac.getTrackName
  api_diagnostics.getTrackName = diagnoseAPI(
    'ac.getTrackName',
    function() return ac.getTrackName end,
    function(func) return func() end
  )

  -- render.createRay
  api_diagnostics.createRay = diagnoseAPI(
    'render.createRay',
    function() return render.createRay end,
    nil  -- Don't call, just check existence
  )

  -- ray:physics (method, check if available)
  api_diagnostics.rayPhysics = diagnoseAPI(
    'ray:physics',
    function()
      -- Create a dummy ray to check if method exists
      local ok, ray = pcall(function() return render.createRay(vec3(0,0,0), vec3(0,1,0)) end)
      if ok and ray then
        return ray.physics
      end
      return nil
    end,
    nil
  )

  -- ray:track (method, check if available)
  api_diagnostics.rayTrack = diagnoseAPI(
    'ray:track',
    function()
      local ok, ray = pcall(function() return render.createRay(vec3(0,0,0), vec3(0,1,0)) end)
      if ok and ray then
        return ray.track
      end
      return nil
    end,
    nil
  )

  -- ===========================================================================
  -- Alternative APIs: candidates to replace missing getTrackSpline / getDocumentsPath / getLayoutName.
  -- A nil result here is NOT treated as a failure: the test input may simply be invalid for the current track.
  -- ===========================================================================

  -- 1. ac.trackProgressToWorldCoordinate(0.5) -> vec3
  --    Candidate replacement for: getTrackSpline (sample the AI spline at fixed progress steps).
  api_diagnostics.alt_trackProgressToWorld = testAlternativeAPI(
    'ac.trackProgressToWorldCoordinate',
    function() return ac.trackProgressToWorldCoordinate end,
    '0.5',
    function(func) return func(0.5) end
  )

  -- 2. ac.worldCoordinateToTrack(vec3(0,0,0)) -> vec3
  --    Inverse of (1). Test point is world origin -- may return nil if far from track, that is expected.
  api_diagnostics.alt_worldToTrack = testAlternativeAPI(
    'ac.worldCoordinateToTrack',
    function() return ac.worldCoordinateToTrack end,
    'vec3(0,0,0) [origin; nil if far from track]',
    function(func) return func(vec3(0, 0, 0)) end
  )

  -- 3. ac.trackCoordinateToWorld(vec3(0,0,0.5)) -> vec3
  --    Track coords: x=normalised lateral (0=centre), y=height above track, z=progress.
  --    This call should always succeed on any loaded track.
  api_diagnostics.alt_trackToWorld = testAlternativeAPI(
    'ac.trackCoordinateToWorld',
    function() return ac.trackCoordinateToWorld end,
    'vec3(0,0,0.5) [centre, on-surface, mid-progress]',
    function(func) return func(vec3(0, 0, 0.5)) end
  )

  -- 4. ac.getTrackAISplineSides(0.5) -> vec2 (left width, right width)
  --    Tells how wide the track is at a given progress. Useful for camera offset planning.
  api_diagnostics.alt_splineSides = testAlternativeAPI(
    'ac.getTrackAISplineSides',
    function() return ac.getTrackAISplineSides end,
    '0.5',
    function(func) return func(0.5) end
  )

  -- 5. ac.getTrackLayout() -> string
  --    Direct replacement candidate for missing ac.getLayoutName().
  api_diagnostics.alt_getTrackLayout = testAlternativeAPI(
    'ac.getTrackLayout',
    function() return ac.getTrackLayout end,
    '(no args)',
    function(func)
      local r = func()
      if r == '' then return '(empty string - no layout)' end
      return r
    end
  )

  -- 6. ac.getFolder(ac.FolderID.ACDocuments) -> string
  --    Direct replacement candidate for missing ac.getDocumentsPath().
  --    Note: ac.FolderID.ACDocuments is an enum (number=31 per SDK). We test the whole chain.
  api_diagnostics.alt_getFolder_ACDocuments = testAlternativeAPI(
    'ac.getFolder(ac.FolderID.ACDocuments)',
    function()
      if ac.getFolder == nil then return nil end
      if ac.FolderID == nil then return nil end
      if ac.FolderID.ACDocuments == nil then return nil end
      return ac.getFolder
    end,
    'FolderID.ACDocuments=' .. tostring(ac.FolderID and ac.FolderID.ACDocuments or 'N/A'),
    function(func) return func(ac.FolderID.ACDocuments) end
  )

  -- Determine overall availability (UNCHANGED: still based on the original four critical APIs)
  local criticalAPIs = {'getTrackSpline', 'getDocumentsPath', 'getLayoutName', 'createRay'}
  local allAvailable = true
  local missingOrFailed = {}

  for _, apiName in ipairs(criticalAPIs) do
    local diag = api_diagnostics[apiName]
    if diag then
      if diag.state == 'FUNCTION_MISSING' then
        allAvailable = false
        missingOrFailed[#missingOrFailed + 1] = apiName .. ' (function missing)'
      elseif diag.state == 'CALL_FAILED' then
        missingOrFailed[#missingOrFailed + 1] = apiName .. ' (call failed)'
      elseif diag.state == 'RETURNED_NIL' then
        missingOrFailed[#missingOrFailed + 1] = apiName .. ' (returned nil)'
      end
    end
  end

  api_available = allAvailable

  if allAvailable then
    scan_error = nil
  else
    scan_error = 'API issues: ' .. table.concat(missingOrFailed, ', ')
  end

  -- Log alternative API summary
  ac.log('=== ALTERNATIVE API SUMMARY ===')
  local altKeys = {
    'alt_trackProgressToWorld', 'alt_worldToTrack', 'alt_trackToWorld',
    'alt_splineSides', 'alt_getTrackLayout', 'alt_getFolder_ACDocuments'
  }
  for _, k in ipairs(altKeys) do
    local d = api_diagnostics[k]
    if d then
      ac.log(string.format('  %-42s %s  %s', d.name, d.state, d.resultStr))
    end
  end
end

--- Minimal raycast test: verify if render.createRay and methods work
local function runRaycastTest()
  raycast_test.enabled = true

  -- Reset results
  raycast_test.ray_creation = nil
  raycast_test.ray_type = nil
  raycast_test.physics_method = nil
  raycast_test.physics_call = nil
  raycast_test.physics_result = nil
  raycast_test.physics_error = nil
  raycast_test.track_method = nil
  raycast_test.track_call = nil
  raycast_test.track_result = nil
  raycast_test.track_error = nil

  ac.log('=== RAYCAST TEST START ===')

  -- 1. Check render.createRay exists
  if render.createRay == nil then
    raycast_test.ray_creation = 'MISSING'
    ac.log('render.createRay: MISSING')
    return
  end

  -- 2. Create a test ray
  local ok, ray = pcall(function()
    -- Test ray from origin pointing down
    return render.createRay(vec3(0, 10, 0), vec3(0, -1, 0))
  end)

  if not ok then
    raycast_test.ray_creation = 'FAIL'
    raycast_test.physics_error = tostring(ray)
    ac.log('render.createRay: FAIL - ' .. tostring(ray))
    return
  end

  if ray == nil then
    raycast_test.ray_creation = 'FAIL'
    raycast_test.physics_error = 'returned nil'
    ac.log('render.createRay: FAIL - returned nil')
    return
  end

  raycast_test.ray_creation = 'SUCCESS'
  raycast_test.ray_type = type(ray)
  ac.log('render.createRay: SUCCESS, type=' .. type(ray))

  -- 3. Check ray:physics method exists
  if ray.physics == nil then
    raycast_test.physics_method = 'MISSING'
    ac.log('ray.physics: MISSING')
  else
    raycast_test.physics_method = 'AVAILABLE'
    ac.log('ray.physics: AVAILABLE')

    -- 4. Call ray:physics()
    local physOk, physResult = pcall(function()
      return ray:physics()
    end)

    if not physOk then
      raycast_test.physics_call = 'FAIL'
      raycast_test.physics_error = tostring(physResult)
      ac.log('ray:physics(): FAIL - ' .. tostring(physResult))
    else
      raycast_test.physics_call = 'SUCCESS'
      raycast_test.physics_result = physResult
      ac.log('ray:physics(): SUCCESS, result=' .. tostring(physResult))
    end
  end

  -- 5. Check ray:track method exists
  if ray.track == nil then
    raycast_test.track_method = 'MISSING'
    ac.log('ray.track: MISSING')
  else
    raycast_test.track_method = 'AVAILABLE'
    ac.log('ray.track: AVAILABLE')

    -- 6. Call ray:track()
    local trackOk, trackResult = pcall(function()
      return ray:track()
    end)

    if not trackOk then
      raycast_test.track_call = 'FAIL'
      raycast_test.track_error = tostring(trackResult)
      ac.log('ray:track(): FAIL - ' .. tostring(trackResult))
    else
      raycast_test.track_call = 'SUCCESS'
      raycast_test.track_result = trackResult
      ac.log('ray:track(): SUCCESS, result=' .. tostring(trackResult))
    end
  end

  -- 7. Cleanup
  pcall(function() ray:dispose() end)

  ac.log('=== RAYCAST TEST END ===')
end

--- Get track spline points from the game.
-- Returns array of vec3 positions forming the racing line.
local function getTrackSpline()
  local track = ac.getTrackSpline()
  if not track then
    return nil, 'ac.getTrackSpline() returned nil'
  end

  local points = {}
  local count = track.splinePointCount or 0
  if count == 0 then
    return nil, 'track spline has no points'
  end

  for i = 0, count - 1 do
    local pt = track:getSplinePoint(i)
    if pt then
      points[#points + 1] = pt.position
    end
  end

  if #points < 2 then
    return nil, 'track spline has fewer than 2 points'
  end

  return points
end

--- Calculate cumulative distance along track spline.
local function calculateCumulativeDistances(points)
  local distances = {0.0}
  for i = 2, #points do
    local dx = points[i].x - points[i-1].x
    local dy = points[i].y - points[i-1].y
    local dz = points[i].z - points[i-1].z
    local dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    distances[i] = distances[i-1] + dist
  end
  return distances
end

--- Find station positions along track at regular intervals.
local function computeStationPositions(points, cumulativeDistances, spacing)
  local totalLength = cumulativeDistances[#cumulativeDistances]
  local stations = {}
  local numStations = math.floor(totalLength / spacing)

  for i = 0, numStations do
    local targetDist = i * spacing
    -- Find segment containing this distance
    local segStart = 1
    for j = 1, #cumulativeDistances - 1 do
      if cumulativeDistances[j+1] >= targetDist then
        segStart = j
        break
      end
    end

    local segEnd = segStart + 1
    if segEnd > #points then break end

    local segLen = cumulativeDistances[segEnd] - cumulativeDistances[segStart]
    if segLen < 1e-6 then
      stations[#stations + 1] = points[segStart]
    else
      local t = (targetDist - cumulativeDistances[segStart]) / segLen
      local p1 = points[segStart]
      local p2 = points[segEnd]
      stations[#stations + 1] = vec3(
        p1.x + (p2.x - p1.x) * t,
        p1.y + (p2.y - p1.y) * t,
        p1.z + (p2.z - p1.z) * t
      )
    end
  end

  return stations
end

--- Compute tangent direction at a station index.
local function computeTangent(stations, idx)
  if idx < #stations then
    local p1 = stations[idx]
    local p2 = stations[idx + 1]
    local dx = p2.x - p1.x
    local dz = p2.z - p1.z
    local len = math.sqrt(dx*dx + dz*dz)
    if len > 1e-6 then
      return dx/len, dz/len
    end
  end
  if idx > 1 then
    local p1 = stations[idx-1]
    local p2 = stations[idx]
    local dx = p2.x - p1.x
    local dz = p2.z - p1.z
    local len = math.sqrt(dx*dx + dz*dz)
    if len > 1e-6 then
      return dx/len, dz/len
    end
  end
  return 0, 1  -- default forward
end

--- Cast a physics ray downward to find ground height.
local function castGroundRay(x, y, z)
  local ray = render.createRay(vec3(x, y + 50, z), vec3(0, -1, 0), RAY_LENGTH)
  local hitPos, hitNormal = ray:physics()
  if hitPos then
    return hitPos.y, hitNormal
  end
  return nil, nil
end

--- Cast a track ray to check sight line to centreline.
local function castSightRay(fromX, fromY, fromZ, toX, toY, toZ)
  local dx = toX - fromX
  local dy = toY - fromY
  local dz = toZ - fromZ
  local len = math.sqrt(dx*dx + dy*dy + dz*dz)
  if len < 1e-6 then return nil, 0 end

  local dir = vec3(dx/len, dy/len, dz/len)
  local ray = render.createRay(vec3(fromX, fromY, fromZ), dir, len)
  local dist = ray:track()
  return dist, len
end

--- Perform one frame of scanning work.
local function scanFrame()
  if not scanning then return end

  local stations = scan_state.stations
  local idx = scan_state.current_cell

  if idx > #scan_state.all_cells then
    scanning = false
    scan_progress = 1.0
    writeScanResult()
    ac.log('=== SCAN COMPLETE ===')
    ac.log('Results written to: ' .. scan_result_file)
    return
  end

  local cell = scan_state.all_cells[idx]
  local stationIdx = cell.station
  local profileIdx = cell.profile

  local station = stations[stationIdx + 1]  -- Lua is 1-based
  local tx, tz = computeTangent(stations, stationIdx + 1)
  -- Perpendicular: rotate 90 degrees
  local px, pz = -tz, tx

  local lateralOffset = (profileIdx - 20) * PROFILE_SPACING  -- 0..40 -> -40..+40
  local worldX = station.x + px * lateralOffset
  local worldZ = station.z + pz * lateralOffset
  local worldY = station.y + 50  -- start ray above everything

  -- Cast ground ray
  local groundHeight, normal = castGroundRay(worldX, worldY, worldZ)

  -- If we got ground, cast sight ray to centreline
  local sightClear = nil
  local sightDistance = nil
  if groundHeight then
    local eyeY = groundHeight + 1.2  -- eye level
    local dist, maxDist = castSightRay(worldX, eyeY, worldZ, station.x, station.y, station.z)
    if dist then
      sightClear = dist >= maxDist - 0.1  -- clear if ray reached target
      sightDistance = dist
    else
      sightClear = false
      sightDistance = 0
    end
  end

  -- Store result
  local quality = groundHeight and 'MEASURED' or 'UNKNOWN'
  scan_grid[#scan_grid + 1] = {
    station_index = stationIdx,
    profile_index = profileIdx,
    position_world = {worldX, groundHeight or worldY, worldZ},
    ground_height_m = groundHeight,
    surface_normal = normal and {normal.x, normal.y, normal.z} or nil,
    sight_clear = sightClear,
    sight_distance_m = sightDistance,
    measurement_quality = quality
  }

  scan_state.current_cell = idx + 1
  scan_progress = idx / #scan_state.all_cells
end

--- Write scan results to JSON file.
local function writeScanResult()
  local json = require('json')

  local data = {
    schema_version = 1,
    track_id = ac.getTrackName() or 'unknown',
    layout = ac.getLayoutName() or '',
    scan_metadata = {
      station_count = scan_state.num_stations,
      profile_points_per_station = 41,  -- -40 to +40 step 2
      station_spacing_m = STATION_SPACING,
      profile_spacing_m = PROFILE_SPACING,
      profile_range_m = 80.0,
      total_cells = #scan_grid,
      csp_version = result.csp,
      scanned_at = os.date('!%Y-%m-%dT%H:%M:%SZ')
    },
    grid = scan_grid
  }

  local content = json.encode(data)

  -- Write to documents folder
  local path = scan_result_file
  local file = io.open(path, 'w')
  if file then
    file:write(content)
    file:close()
  else
    ac.log('ERROR: Could not write to ' .. path)
  end
end

--- Start the terrain scan.
local function startScan()
  if scanning then
    ac.log('Scan already in progress')
    return
  end

  ac.log('=== STARTING TERRAIN SCAN ===')

  -- Get track spline
  local points, err = getTrackSpline()
  if not points then
    ac.log('ERROR: ' .. err)
    scan_error = err
    return
  end

  -- Calculate stations
  local cumulativeDistances = calculateCumulativeDistances(points)
  local stations = computeStationPositions(points, cumulativeDistances, STATION_SPACING)

  scan_state = {
    stations = stations,
    num_stations = #stations,
    current_cell = 1,
    all_cells = {}
  }

  -- Generate all cells
  local numProfiles = 41  -- -40 to +40 step 2
  for s = 0, #stations - 1 do
    for p = 0, numProfiles - 1 do
      scan_state.all_cells[#scan_state.all_cells + 1] = {
        station = s,
        profile = p
      }
    end
  end

  scan_total_stations = #stations
  scan_current_station = 0
  scan_grid = {}
  scanning = true
  scan_progress = 0
  scan_error = nil

  -- Generate output filename
  local trackName = ac.getTrackName() or 'unknown_track'
  local layoutName = ac.getLayoutName() or ''
  scan_result_file = string.format('%s/%s_%s_terrain_scan.json',
    ac.getDocumentsPath(),
    trackName,
    layoutName ~= '' and layoutName or 'main'
  )

  ac.log(string.format('Scanning %d stations x %d profiles = %d cells',
    #stations, numProfiles, #scan_state.all_cells))
end

--- The app window. Draws the result and nothing else.
function script.windowMain()
  if not probed then
    probed = true
    runProbes()
    runFullAPIDiagnostic()
  end

  -- Run scan frame if active
  if scanning then
    -- Do multiple cells per frame for speed
    for i = 1, 100 do
      scanFrame()
      if not scanning then break end
    end
  end

  ui.text('=== CSP CAPABILITIES ===')
  ui.text('csp     : ' .. result.csp)
  ui.text('replay  : ' .. result.replay)
  ui.text('grab    : ' .. result.grab)
  ui.text('clock   : ' .. result.clock)
  ui.separator()

  ui.text('=== TERRAIN SCAN ===')

  if not api_available then
    ui.text('STATUS: API DIAGNOSTIC')
    ui.separator()

    -- Show detailed diagnostic for each API
    if api_diagnostics.getTrackSpline then
      local diag = api_diagnostics.getTrackSpline
      ui.text(string.format('ac.getTrackSpline: %s', diag.state))
      ui.text(string.format('  exists=%s, type=%s, result=%s',
        tostring(diag.exists), diag.type, diag.resultType))
      if diag.callError then
        ui.text('  error: ' .. diag.callError)
      end
    end

    if api_diagnostics.getDocumentsPath then
      local diag = api_diagnostics.getDocumentsPath
      ui.text(string.format('ac.getDocumentsPath: %s', diag.state))
      ui.text(string.format('  exists=%s, type=%s, result=%s',
        tostring(diag.exists), diag.type, diag.resultType))
      if diag.callError then
        ui.text('  error: ' .. diag.callError)
      end
    end

    if api_diagnostics.getLayoutName then
      local diag = api_diagnostics.getLayoutName
      ui.text(string.format('ac.getLayoutName: %s', diag.state))
      ui.text(string.format('  exists=%s, type=%s, result=%s',
        tostring(diag.exists), diag.type, diag.resultType))
      if diag.callError then
        ui.text('  error: ' .. diag.callError)
      end
    end

    if api_diagnostics.createRay then
      local diag = api_diagnostics.createRay
      ui.text(string.format('render.createRay: %s', diag.state))
      ui.text(string.format('  exists=%s, type=%s',
        tostring(diag.exists), diag.type))
      if diag.callError then
        ui.text('  error: ' .. diag.callError)
      end
    end

    ui.separator()
    if scan_error then
      ui.textWrapped('Summary: ' .. scan_error)
    end
    ui.textWrapped('Press R to re-run diagnostic after track loads.')

  elseif scanning then
    ui.text(string.format('Progress: %.1f%% (%d/%d cells)',
      scan_progress * 100,
      scan_state.current_cell - 1,
      #scan_state.all_cells
    ))
    if ui.button('Stop Scan') then
      scanning = false
    end
  else
    if ui.button('Start Terrain Scan') then
      startScan()
    end
    if scan_error then
      ui.text('Last scan: FAILED')
      ui.textWrapped('Error: ' .. scan_error)
    elseif scan_progress > 0 and scan_progress < 1.0 then
      ui.text('Last scan: interrupted')
    elseif scan_progress >= 1.0 then
      ui.text('Last scan: complete')
      ui.text('File: ' .. scan_result_file)
    end
  end

  -- Alternative API tests (replacement candidates for missing APIs).
  -- Always shown once the diagnostic has run, regardless of api_available.
  local altKeys = {
    'alt_trackProgressToWorld', 'alt_worldToTrack', 'alt_trackToWorld',
    'alt_splineSides', 'alt_getTrackLayout', 'alt_getFolder_ACDocuments'
  }
  local hasAltResults = false
  for _, k in ipairs(altKeys) do
    if api_diagnostics[k] then hasAltResults = true break end
  end

  if hasAltResults then
    ui.separator()
    ui.text('=== ALTERNATIVE API TESTS ===')
    ui.textWrapped('Candidates to replace missing getTrackSpline / getDocumentsPath / getLayoutName.')
    ui.textWrapped('NOTE: "returned nil" is NOT a failure -- the test input may be invalid for this track.')
    for _, k in ipairs(altKeys) do
      local d = api_diagnostics[k]
      if d then
        ui.text(string.format('%s: %s', d.name, d.state))
        ui.text(string.format('  test: %s', d.testArgs))
        ui.text(string.format('  -> %s', d.resultStr))
      end
    end
  end

  -- Re-run diagnostic button (for debugging)
  if ui.button('Re-run API Diagnostic (R)') then
    runFullAPIDiagnostic()
    ac.log('API diagnostic re-run completed')
  end

  -- Raycast test button
  if ui.button('Test Raycast') then
    runRaycastTest()
  end

  -- Display raycast test results
  if raycast_test.enabled then
    ui.separator()
    ui.text('=== RAYCAST TEST ===')
    ui.text(string.format('Ray creation: %s', raycast_test.ray_creation or 'not run'))

    if raycast_test.ray_creation == 'SUCCESS' then
      ui.text(string.format('Ray type: %s', raycast_test.ray_type or 'unknown'))
      ui.text(string.format('ray.physics: %s', raycast_test.physics_method or 'not checked'))

      if raycast_test.physics_call then
        ui.text(string.format('  ray:physics(): %s', raycast_test.physics_call))
        if raycast_test.physics_error then
          ui.text(string.format('  Error: %s', raycast_test.physics_error))
        elseif raycast_test.physics_result ~= nil then
          ui.text(string.format('  Result: %s', tostring(raycast_test.physics_result)))
        end
      end

      ui.text(string.format('ray.track: %s', raycast_test.track_method or 'not checked'))

      if raycast_test.track_call then
        ui.text(string.format('  ray:track(): %s', raycast_test.track_call))
        if raycast_test.track_error then
          ui.text(string.format('  Error: %s', raycast_test.track_error))
        elseif raycast_test.track_result ~= nil then
          ui.text(string.format('  Result: %s', tostring(raycast_test.track_result)))
        end
      end
    end
  end

  ui.separator()
  ui.textWrapped('Copy the four capability lines back. For terrain scan, send the .json file.')
end

-- Initialize scan state
scan_state = nil
