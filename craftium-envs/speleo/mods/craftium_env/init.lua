voxel_radius = {
	x = minetest.settings:get("voxel_obs_rx"),
	y = minetest.settings:get("voxel_obs_ry"),
	z = minetest.settings:get("voxel_obs_rz")
}

-- Executed when the player joins the game
minetest.register_on_joinplayer(function(player, _last_login)
	-- Set the players initial position
	player:set_pos({x = 24.3, y = 5.5, z=-36.3})

	-- Disable HUD elements
	player:hud_set_flags({
		hotbar = false,
		crosshair = false,
		healthbar = false,
	})
end)

minetest.register_globalstep(function(dtime)
	local player = minetest.get_connected_players()[1]

	-- if the player is not connected end here
	if player == nil then
		return nil
	end

	-- if the player is connected:
	local player_pos = player:get_pos()
	if minetest.settings:get("voxel_obs") then
		local voxel_data, voxel_light_data, voxel_param2_data = voxel_api:get_voxel_data(player_pos, voxel_radius)
		set_voxel_data(voxel_data)
		set_voxel_light_data(voxel_light_data)
		set_voxel_param2_data(voxel_param2_data)
	end

	-- set the reward to the inverse of the player's
	-- position on the Y axis (depth)
	set_reward(-player_pos.y)
end)

minetest.register_on_dieplayer(function(_player, _reason)
	-- End episode if the player dies
	set_termination()
end)
