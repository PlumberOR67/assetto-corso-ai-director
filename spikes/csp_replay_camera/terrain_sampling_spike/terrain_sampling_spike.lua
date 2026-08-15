-- Terrain Sampling Spike v2
--
-- Proves the sampling pipeline that the Terrain Scanner will be rebuilt around,
-- using only APIs confirmed present in CSP 0.2.11 build 3465:
--
--   ac.trackProgressToWorldCoordinate(progress)  -> vec3  (centre of track)
--   ac.getTrackAISplineSides(progress)           -> vec2  (x=left width, y=right width)
--   ac.trackCoordinateToWorld(vec3)              -> vec3  (track coords -> world)
--   ac.getFolder(ac.FolderID.ACDocuments)        -> path  (output location)
--   render.createRay + ray:physics(outPos, outNormal) -> raycast down to find surface
--
-- For every station along the lap (progress 0..1 in fixed steps) it:
--   1. samples the centreline world position
--   2. reads the track width from the AI spline sides
--   3. samples the left/right track boundaries via track coordinates
--   4. raycasts straight down at centre, left and right to find the surface
--   5. counts hits vs misses
-- Then writes a small JSON to the Assetto Corsa Documents folder and reports
-- the hit/miss statistics in the window.
--
-- This is deliberately a SEPARATE app. The existing Terrain Scanner
-- (startScan/scanFrame/castGroundRay/castSightRay) is left untouched until this
-- pipeline is proven on a real replay.
--
-- Raycast convention (from the CSP SDK):
--   ray:physics(outPosition, outNormal) returns a NUMBER -- intersection
--   distance, or -1 if there was no intersection. The hit position and normal
--   are written into the out-parameter vec3s. We still fall back to computing
--   the hit position from origin+dir*distance in case a build returns the hit
--   position directly instead.

local STEP = 0.01        -- progress step: 101 stations across the lap
local RAISE = 50.0       -- metres to raise ray origins above the surface
local RAY_LENGTH = 200.0 -- metres, enough to reach the ground from the raised origin
local STATIONS_PER_FRAME = 10  -- keep the window responsive during sampling

local state = {
  probed = false,
  csp = 'unknown',
  running = false,
  done = 0,
  total = 0,
  hits = 0,
  misses = 0,
  samples = {},
  output_file = '',
  error = nil,
  last_stats = nil,
}

--- Raycast straight down from `from`. Returns the ground hit point (vec3) and
-- distance, or nil when there is no hit.
local function castDown(from)
  local ok, ray = pcall(function()
    return render.createRay(vec3(from.x, from.y + RAISE, from.z), vec3(0, -1, 0), RAY_LENGTH)
  end)
  if not ok or ray == nil then return nil end

  -- Sentinel that is indistinguishable from any real hit position.
  local outPos = vec3(9e9, 9e9, 9e9)
  local outNormal = vec3(0, 0, 0)
  local physOk, first, second = pcall(function()
    return ray:physics(outPos, outNormal)
  end)
  pcall(function() ray:dispose() end)

  if not physOk then return nil end

  if type(first) == 'number' then
    -- Documented convention: first = distance, -1 means no hit.
    if first < 0 then return nil end
    local hit
    if outPos.x ~= 9e9 or outPos.y ~= 9e9 or outPos.z ~= 9e9 then
      hit = outPos
    else
      -- Out-parameter was not written to; reconstruct the hit point from the ray.
      hit = vec3(from.x, from.y + RAISE - first, from.z)
    end
    return hit, first
  elseif first ~= nil then
    -- Defensive: some builds may return the hit position directly instead.
    return first, second
  end

  return nil
end

--- Sample one station. Records the result into state.samples.
local function sampleStation(progress)
  local center = ac.trackProgressToWorldCoordinate(progress)
  local sides = ac.getTrackAISplineSides(progress)
  local left = ac.trackCoordinateToWorld(vec3(-1, 0, progress))
  local right = ac.trackCoordinateToWorld(vec3(1, 0, progress))

  local sample = {
    progress = progress,
    center = center and { x = center.x, y = center.y, z = center.z } or nil,
    left_width_m = sides and sides.x or nil,
    right_width_m = sides and sides.y or nil,
    rays = {},
  }

  -- Cast down at the three track points.
  local points = {
    { key = 'center', pos = center },
    { key = 'left',   pos = left },
    { key = 'right',  pos = right },
  }
  for _, p in ipairs(points) do
    if p.pos then
      local hit, dist = castDown(p.pos)
      if hit then
        state.hits = state.hits + 1
        sample.rays[#sample.rays + 1] = {
          point = p.key,
          hit = true,
          distance_m = dist,
          ground_y = hit.y,
          world = { x = hit.x, y = hit.y, z = hit.z },
        }
      else
        state.misses = state.misses + 1
        sample.rays[#sample.rays + 1] = {
          point = p.key,
          hit = false,
          distance_m = nil,
        }
      end
    else
      state.misses = state.misses + 1
      sample.rays[#sample.rays + 1] = {
        point = p.key,
        hit = false,
        error = 'no track point',
      }
    end
  end

  state.samples[#state.samples + 1] = sample
  state.done = state.done + 1
end

--- Write the JSON file and compute final statistics.
local function finishSampling()
  state.running = false

  state.last_stats = {
    total_stations = state.total,
    total_rays = state.total * 3,
    hits = state.hits,
    misses = state.misses,
    hit_rate = state.hits + state.misses > 0 and state.hits / (state.hits + state.misses) or 0,
  }

  ac.log('=== TERRAIN SAMPLING SPIKE V2: COMPLETE ===')
  ac.log(string.format('stations=%d rays=%d hits=%d misses=%d rate=%.1f%%',
    state.last_stats.total_stations,
    state.last_stats.total_rays,
    state.last_stats.hits,
    state.last_stats.misses,
    state.last_stats.hit_rate * 100))

  local ok, res = pcall(function()
    local json = require('json')
    local data = {
      schema_version = 2,
      app = 'terrain_sampling_spike_v2',
      csp_version = state.csp,
      sampling = {
        step = STEP,
        total_stations = state.total,
        total_rays = state.total * 3,
        hits = state.hits,
        misses = state.misses,
      },
      samples = state.samples,
    }
    local path = ac.getFolder(ac.FolderID.ACDocuments) .. '/terrain_sampling_spike.json'
    local file = io.open(path, 'w')
    if file then
      file:write(json.encode(data))
      file:close()
      return path
    end
    return nil, 'could not open output file: ' .. path
  end)

  if ok and res then
    state.output_file = res
    ac.log('JSON saved: ' .. res)
  else
    state.error = tostring(res)
    ac.log('ERROR: ' .. tostring(res))
  end
end

--- Begin sampling a full lap.
local function startSampling()
  if state.running then return end

  state.running = true
  state.done = 0
  state.hits = 0
  state.misses = 0
  state.samples = {}
  state.error = nil
  state.output_file = ''
  state.last_stats = nil

  -- 101 stations: progress = idx * STEP for idx in 0..100 (0.0 to 1.0 inclusive).
  state.total = math.floor(1.0 / STEP) + 1

  ac.log('=== TERRAIN SAMPLING SPIKE V2: START ===')
  ac.log(string.format('stations=%d step=%.2f csp=%s', state.total, STEP, state.csp))
end

--- One-time startup checks: report the patch version.
local function runStartupChecks()
  local ok, version = pcall(function() return ac.getPatchVersion() end)
  state.csp = ok and tostring(version) or 'unknown'
end

--- Draw the app window.
local function drawWindow()
  ui.text('=== TERRAIN SAMPLING SPIKE V2 ===')
  ui.text('csp: ' .. state.csp)

  if not state.running then
    if ui.button('Start Sampling') then
      startSampling()
    end
  else
    ui.text(string.format('Sampling... %d/%d stations', state.done, state.total))
    if ui.button('Stop') then
      state.running = false
    end
  end

  if state.error then
    ui.separator()
    ui.textWrapped('Error: ' .. state.error)
  end

  if state.last_stats then
    ui.separator()
    ui.text('=== STATISTICS ===')
    ui.text(string.format('Stations : %d', state.last_stats.total_stations))
    ui.text(string.format('Rays     : %d (%d per station)', state.last_stats.total_rays, 3))
    ui.text(string.format('Hits     : %d', state.last_stats.hits))
    ui.text(string.format('Misses   : %d', state.last_stats.misses))
    ui.text(string.format('Hit rate : %.1f%%', state.last_stats.hit_rate * 100))
    if state.output_file ~= '' then
      ui.separator()
      ui.text('Saved: ' .. state.output_file)
    end
  end
end

--- The app window. Runs the sampling a few stations per frame.
function script.windowMain()
  if not state.probed then
    state.probed = true
    runStartupChecks()
  end

  if state.running then
    local target = math.min(state.total, state.done + STATIONS_PER_FRAME)
    while state.done < target do
      sampleStation(state.done * STEP)
    end
    if state.done >= state.total then
      finishSampling()
    end
  end

  drawWindow()
end
