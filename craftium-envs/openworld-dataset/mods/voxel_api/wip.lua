-- Function to check if a player is stuck in terrain
local function check_player_clip(player)
    if not player then return false end

    -- Get player's position and collision box
    local pos = player:get_pos()
    local player_box = {
        x = 0.3,  -- Half width of player
        y = 0.9,  -- Half height of player (assuming 1.8 blocks tall)
        z = 0.3   -- Half depth of player
    }

    -- Check surrounding nodes for collision
    for x = -player_box.x, player_box.x do
        for y = 0, player_box.y * 2 do  -- Check full height
            for z = -player_box.z, player_box.z do
                local check_pos = {
                    x = math.floor(pos.x + x),
                    y = math.floor(pos.y + y),
                    z = math.floor(pos.z + z)
                }

                local node = minetest.get_node(check_pos)
                local node_def = minetest.registered_nodes[node.name]

                -- Check if node is solid and walkable
                if node_def and node_def.walkable then
                    -- Check if player's collision box intersects with node
                    local player_min = {
                        x = pos.x - player_box.x,
                        y = pos.y,
                        z = pos.z - player_box.z
                    }
                    local player_max = {
                        x = pos.x + player_box.x,
                        y = pos.y + (player_box.y * 2),
                        z = pos.z + player_box.z
                    }

                    -- If player box intersects with solid node, they're clipping
                    if player_min.x < check_pos.x + 1 and player_max.x > check_pos.x and
                       player_min.y < check_pos.y + 1 and player_max.y > check_pos.y and
                       player_min.z < check_pos.z + 1 and player_max.z > check_pos.z then
                        return true
                    end
                end
            end
        end
    end

    return false
end

-- Register a globalstep to continuously check for clipping
minetest.register_globalstep(function(dtime)
    for _, player in ipairs(minetest.get_connected_players()) do
        if check_player_clip(player) then
            local name = player:get_player_name()
            minetest.chat_send_player(name, "Warning: You are clipping through terrain!")

            -- Optional: Move player up until they're not clipping
            local pos = player:get_pos()
            while check_player_clip(player) do
                pos.y = pos.y + 1
                player:set_pos(pos)
            end
        end
    end
end)

minetest.register_chatcommand("check_clip", {
    description = "Check if a player is clipping through terrain",
    params = "[player_name]", -- Optional parameter
    privs = {teleport = true}, -- Require teleport privilege to use command

    func = function(name, param)
        local player
        if param and param ~= "" then
            -- Check specified player
            player = minetest.get_player_by_name(param)
            if not player then
                return false, "Player " .. param .. " not found"
            end
        else
            -- Check command issuer
            player = minetest.get_player_by_name(name)
        end

        if check_player_clip(player) then
            return true, "Player is clipping through terrain!"
        else
            return true, "Player is not clipping"
        end
    end
})
