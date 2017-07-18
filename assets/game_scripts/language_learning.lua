local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local screen_message = require 'common.screen_message'
local tensor = require 'dmlab.system.tensor'
local random = require 'common.random'
local custom_observations = require 'decorators.custom_observations'

local api = {}

local entityLayer = [[
**************
*            *
*            *
*            *
*****    *****
*****    *****
*****    *****
*            *
*            *
*            *
*          P *
**************
]]

local variationsLayer = [[
..............
..............
..............
..............
..............
..............
..............
..............
..............
..............
]]

random.seed(32)
------------------------------------------------------------------
local object_types = {'A', 'F', 'L', 'S'}
local layout_types = {'A', 'B', 'C'}
local object_map = {['A'] = 'apple', ['L'] = 'lemon', 
                    ['F'] = 'fungi', ['S'] = 'strawberry'}
local word2id = {['find']=0, ['apple']=1, ['fungi']=2, ['lemon']=3, 
                 ['strawberry']=4, ['pad']=5, ['start']=6, ['stop']=7}
local id2word = {[0] = 'find', [1] = 'apple', [2] = 'fungi', [3] = 'lemon', 
                 [4] = 'strawberry', [5] = 'pad', [6] = 'start', [7] = 'stop'}

-----------------------------------------------------------------
-- Build the map with random positioning of objects
local present_objects = {}

for i = 1, #entityLayer do
  if (entityLayer:sub(i, i) == ' ' and random.uniformReal(0, 1) < 0.4) then
    local object_to_pick = random.uniformInt(1, 4)
    entityLayer = entityLayer:sub(1, i-1) .. object_types[object_to_pick] .. entityLayer:sub(i+1)

    -- Include present objects
    if present_objects[object_to_pick] == nil then 
      present_objects[object_to_pick] = 'Yay'
    end    
  end
end 

-- Build the command in text and index encoded
local command = {}
local keyset={}
-- Build list of present objects
for k,v in pairs(present_objects  ) do
  table.insert(keyset, k)
end
-- Choose a random object type from the present objects
local object_to_find = random.uniformInt(1, #keyset)
command['command'] = tensor.Tensor{6, 0, keyset[object_to_find], 7}
command['text'] = 'find ' .. id2word[keyset[object_to_find]]

-- Set the score of the game based on the chosen object
-- Chosen object has positive score, whereas every other object types have negative score
local object = object_types[keyset[object_to_find]]
local reward = ''
local rewards = {'apple_reward', 'fungi_reward', 'strawberry_reward', 'lemon_reward'}

if object == 'A' then 
  reward = 'apple_reward'
elseif object == 'F' then 
  reward = 'fungi_reward'
elseif object == 'S' then 
  reward = 'strawberry_reward'
elseif object == 'L' then 
  reward = 'lemon_reward'
end

for i = 1, 4 do 
  if pickups.defaults[rewards[i]]['class_name'] == reward then 
    pickups.defaults[rewards[i]]['quantity'] = 1
  else 
    pickups.defaults[rewards[i]]['quantity'] = -1
  end
end
-------------------------------------------------------------------
print('LEVEL SHAPE')
print(entityLayer)

print('Reward object', reward)


function api:start(episode, seed)
  make_map.seedRng(seed)
  api._count = 0
end

local observationTable = {
    ORDER = command.command,
}

function api:customObservationSpec()
  return {
    {name = 'ORDER', type = 'Doubles', shape = {4}},
  }
end

function api:customObservation(name)
  return observationTable[name]
end

function api:commandLine(oldCommandLine)
  return make_map.commandLine(oldCommandLine)
end 

function api:createPickup(className)
  return pickups.defaults[className]
end

function api:nextMap()
  api._count = api._count + 1
  return make_map.makeMap("demo_map_" .. api._count, entityLayer, variationsLayer)
end

function api:screenMessages(args)
  local message_order = {
      message = command.text,
      x = 0,
      y = 0,
      alignment = screen_message.ALIGN_LEFT,
  }
  return { message_order }
end


return api
