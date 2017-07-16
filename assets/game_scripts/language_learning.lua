local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local api = {}


local entityLayer = [[
**************
*  G     E  R*
*  N     I  O*
*  B      D  *
*****    *****
*****    *****
*****    *****
*            *
*            *
*            *
* A B C    P *
**************
]]

local variationsLayer = [[
..............
..............
........AAA...
..............
..............
..............
..............
..............
.CCCCCCCC.BBB.
..............
]]


function api:start(episode, seed)
  make_map.seedRng(seed)
  api._count = 0
end

function api:commandLine(oldCommandLine)
  return make_map.commandLine(oldCommandLine)
end

function api:createPickup(className)
  return pickups.defaults[className]
end

function api:nextMap()
  map = "G I A P"
  api._count = api._count + 1
  for i = 0, api._count do
    map = map.." A"
  end
  return make_map.makeMap("demo_map_" .. api._count, entityLayer, variationsLayer)
end

return api
