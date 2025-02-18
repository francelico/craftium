-- Set the random seed
if minetest.settings:has("fixed_map_seed") then
	math.randomseed(minetest.settings:get("fixed_map_seed"))
end

function rand(lower, greater)
	return lower + math.random()  * (greater - lower);
end

voxel_radius = {
	x = minetest.settings:get("voxel_obs_rx"),
	y = minetest.settings:get("voxel_obs_ry"),
	z = minetest.settings:get("voxel_obs_rz")
}

north_wall_center = {x = -4, z = -2, y = 5.5} -- 10 blocks in front of player
west_wall_center = {x = -12, z = -12, y = 5.5} -- 8 blocks to the left of player
east_wall_center = {x = 3, z = -12, y = 5.5} -- 7 blocks to the right of player
player_pos = {x = 0, z = -6, y = 5.5}

reset_environment = function()
	local player = minetest.get_connected_players()[1]
	-- Room environment:
	--
	-- ________< (x=4.2, z=-0.8)
	-- | b    |
	-- |_ _ _ |  --> Box spawn area
	-- |      |  --> Agent's spawn area
	-- |  a   |
	-- |______|< (x=4.2, z=-24.2)
	-- ^(x=-13.2, z=-24.2

	--- Spawn a red block inside the room in a random position
	for wall_length = -5, 5 do
		for wall_height = 0, 4 do
			block_pos_northwall = table.copy(north_wall_center)
			block_pos_northwall.x = block_pos_northwall.x + wall_length
			block_pos_northwall.y = block_pos_northwall.y + wall_height
			minetest.set_node(block_pos_northwall, { name = "default:coral_orange" })
			block_pos_eastwall = table.copy(east_wall_center)
			block_pos_eastwall.z = block_pos_eastwall.z + wall_length
			block_pos_eastwall.y = block_pos_eastwall.y + wall_height
			minetest.set_node(block_pos_eastwall, { name = "default:sand" })
			block_pos_westwall = table.copy(west_wall_center)
			block_pos_westwall.z = block_pos_westwall.z + wall_length
			block_pos_westwall.y = block_pos_westwall.y + wall_height
			minetest.set_node(block_pos_westwall, { name = "default:pine_tree" })
		end
	end

	-- Place the player in a fixed location in the middle of the room
	player:set_pos(player_pos)
	eye_offset = player:get_eye_offset()
	print("eye offset (NUE): ", eye_offset.x, eye_offset.y, eye_offset.z)

	-- Disable HUD elements
	player:hud_set_flags({
		hotbar = false,
		crosshair = false,
		healthbar = false,
	})
end

-- Executed when the player joins the game
minetest.register_on_joinplayer(function(player, _last_login)
	reset_environment()
end)

minetest.register_globalstep(function(dtime)
	-- set timeofday to midday
	minetest.set_timeofday(0.5)

	-- get the first connected player
	local player = minetest.get_connected_players()[1]

	-- if the player is not connected end here
	if player == nil then
		return nil
	end

	-- Reset the environment if requested by the python interface
	if get_soft_reset() == 1 then
		reset_environment()
		reset_termination()
	end

	-- set the player's view to the next yaw
	player:set_pos(player_pos)
	local true_player_pos = player:get_pos()
	local eye_height = player:get_properties().eye_height
	local eye_offset = player:get_eye_offset()
	print("playerpos (NUE): ", true_player_pos.x, true_player_pos.y, true_player_pos.z)
	print("eye_height (NUE): ", eye_height)
	print("eye offset (NUE): ", eye_offset.x, eye_offset.y, eye_offset.z)
	local YAW = math.random(0, 360)
	local PITCH = math.random(-20, 20)
	player:set_look_vertical(math.rad(PITCH))
	player:set_look_horizontal(math.rad(YAW))

	local player_pos = player:get_pos()
	if minetest.settings:get("voxel_obs") then
		local voxel_data, voxel_light_data, voxel_param2_data = voxel_api:get_voxel_data(player_pos, voxel_radius)
		set_voxel_data(voxel_data)
		set_voxel_light_data(voxel_light_data)
		set_voxel_param2_data(voxel_param2_data)
	end

	-- the reward at each timestep is -1
	set_reward(-1.0)

end)
