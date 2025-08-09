// Luanti
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2013 celeron55, Perttu Ahola <celeron55@gmail.com>

#include "scripting_server.h"
#include "server.h"
#include "log.h"
#include "settings.h"
#include "filesys.h"
#include "cpp_api/s_internal.h"
#include "lua_api/l_areastore.h"
#include "lua_api/l_auth.h"
#include "lua_api/l_base.h"
#include "lua_api/l_craft.h"
#include "lua_api/l_env.h"
#include "lua_api/l_inventory.h"
#include "lua_api/l_item.h"
#include "lua_api/l_itemstackmeta.h"
#include "lua_api/l_mapgen.h"
#include "lua_api/l_modchannels.h"
#include "lua_api/l_nodemeta.h"
#include "lua_api/l_nodetimer.h"
#include "lua_api/l_noise.h"
#include "lua_api/l_object.h"
#include "lua_api/l_playermeta.h"
#include "lua_api/l_particles.h"
#include "lua_api/l_rollback.h"
#include "lua_api/l_server.h"
#include "lua_api/l_util.h"
#include "lua_api/l_vmanip.h"
#include "lua_api/l_settings.h"
#include "lua_api/l_http.h"
#include "lua_api/l_storage.h"
#include "lua_api/l_ipc.h"
#include "common/c_converter.h"
#include "common/c_content.h"
#include "serverenvironment.h"

extern "C" {
#include <lualib.h>
}

#include "../client/craftium.h"

// Implementation of the C++ voxel data function
// todo: make a version of this function that is directly in client.cpp
inline static int lua_get_voxel_data_cpp(lua_State* L) {
	try {
		// Get function arguments: player_pos (table), radius (table)
		if (!lua_istable(L, 1) || !lua_istable(L, 2)) {
			throw std::runtime_error("Expected table arguments for player_pos and radius");
		}

		// Extract player position
		lua_getfield(L, 1, "x");
		lua_getfield(L, 1, "y");
		lua_getfield(L, 1, "z");
		v3f player_pos(lua_tonumber(L, -3), lua_tonumber(L, -2), lua_tonumber(L, -1));
		lua_pop(L, 3);

		// Extract radius
		lua_getfield(L, 2, "x");
		lua_getfield(L, 2, "y");
		lua_getfield(L, 2, "z");
		v3f radius(lua_tonumber(L, -3), lua_tonumber(L, -2), lua_tonumber(L, -1));
		lua_pop(L, 3);

		// Get the server environment
		Environment* env = ModApiBase::getEnv(L);
		if (!env) {
			throw std::runtime_error("Could not get server environment");
		}

		ServerEnvironment* server_env = dynamic_cast<ServerEnvironment*>(env);
		if (!server_env) {
			throw std::runtime_error("Environment is not a server environment");
		}

		// Calculate bounds
		v3s16 p1 = floatToInt(player_pos - radius, 1);
		v3s16 p2 = floatToInt(player_pos + radius, 1);

		// Create VoxelManip and read from map
		MMVManip vm(&server_env->getServerMap());
		vm.initialEmerge(getNodeBlockPos(p1), getNodeBlockPos(p2));

		// Get both data arrays atomically from the same VoxelManip instance
		const u32 volume = vm.m_area.getVolume();
		std::vector<uint32_t> voxel_data;
		std::vector<uint32_t> voxel_param2_data;

		voxel_data.reserve(volume);
		voxel_param2_data.reserve(volume);

		// Single loop to extract both data types atomically
		for (u32 i = 0; i < volume; i++) {
			if (vm.m_flags[i] & VOXELFLAG_NO_DATA) {
				voxel_data.push_back(CONTENT_IGNORE);
				voxel_param2_data.push_back(0);
			} else {
				voxel_data.push_back(vm.m_data[i].getContent());
				voxel_param2_data.push_back(vm.m_data[i].getParam2());
			}
		}

		// Store in global variables atomically
		g_voxel_data = std::move(voxel_data);
		g_voxel_param2_data = std::move(voxel_param2_data);
		g_voxel_center = floatToInt(player_pos, 1);

		return 0;
	} catch (const std::exception& e) {
		lua_pushstring(L, e.what());
		lua_error(L);
		return 0;
	}
}

ServerScripting::ServerScripting(Server* server):
		ScriptApiBase(ScriptingType::Server),
		asyncEngine(server)
{
	setGameDef(server);

	// setEnv(env) is called by ScriptApiEnv::initializeEnvironment()
	// once the environment has been created

	SCRIPTAPI_PRECHECKHEADER

	if (g_settings->getBool("secure.enable_security")) {
		initializeSecurity();
	} else {
		warningstream << "\\!/ Mod security should never be disabled, as it allows any mod to "
				<< "access the host machine."
				<< "Mods should use minetest.request_insecure_environment() instead \\!/" << std::endl;
	}

	lua_getglobal(L, "core");
	int top = lua_gettop(L);

	lua_newtable(L);
	lua_setfield(L, -2, "object_refs");

	lua_newtable(L);
	lua_setfield(L, -2, "luaentities");

        /* Functions to get/set the global reward and termination values */
        lua_register(L, "set_reward", lua_set_reward);
        lua_register(L, "set_reward_once", lua_set_reward_once);
        lua_register(L, "get_reward", lua_get_reward);
        lua_register(L, "set_termination", lua_set_termination);
        lua_register(L, "reset_termination", lua_reset_termination);
        lua_register(L, "get_termination", lua_get_termination);
        lua_register(L, "get_soft_reset", lua_get_soft_reset);
        lua_register(L, "set_voxel_data", lua_set_voxel_data);
        lua_register(L, "set_voxel_light_data", lua_set_voxel_light_data);
        lua_register(L, "set_voxel_param2_data", lua_set_voxel_param2_data);
        lua_register(L, "get_voxel_data_cpp", lua_get_voxel_data_cpp);

	// Initialize our lua_api modules
	InitializeModApi(L, top);
	lua_pop(L, 1);

	// Push builtin initialization type
	lua_pushstring(L, "game");
	lua_setglobal(L, "INIT");

	infostream << "SCRIPTAPI: Initialized game modules" << std::endl;
}

void ServerScripting::loadBuiltin()
{
	loadMod(Server::getBuiltinLuaPath() + DIR_DELIM "init.lua", BUILTIN_MOD_NAME);
	checkSetByBuiltin();
}

void ServerScripting::saveGlobals()
{
	SCRIPTAPI_PRECHECKHEADER

	lua_getglobal(L, "core");
	luaL_checktype(L, -1, LUA_TTABLE);
	lua_getfield(L, -1, "get_globals_to_transfer");
	lua_call(L, 0, 1);
	auto *data = script_pack(L, -1);
	assert(!data->contains_userdata);
	getServer()->m_lua_globals_data.reset(data);
	// unset the function
	lua_pushnil(L);
	lua_setfield(L, -3, "get_globals_to_transfer");
	lua_pop(L, 2); // pop 'core', return value
}

void ServerScripting::initAsync()
{
	infostream << "SCRIPTAPI: Initializing async engine" << std::endl;
	asyncEngine.registerStateInitializer(InitializeAsync);
	asyncEngine.registerStateInitializer(ModApiUtil::InitializeAsync);
	asyncEngine.registerStateInitializer(ModApiCraft::InitializeAsync);
	asyncEngine.registerStateInitializer(ModApiItem::InitializeAsync);
	asyncEngine.registerStateInitializer(ModApiServer::InitializeAsync);
	asyncEngine.registerStateInitializer(ModApiIPC::Initialize);
	// not added: ModApiMapgen is a minefield for thread safety
	// not added: ModApiHttp async api can't really work together with our jobs
	// not added: ModApiStorage is probably not thread safe(?)

	asyncEngine.initialize(0);
}

void ServerScripting::stepAsync()
{
	asyncEngine.step(getStack());
}

u32 ServerScripting::queueAsync(std::string &&serialized_func,
	PackedValue *param, const std::string &mod_origin)
{
	return asyncEngine.queueAsyncJob(std::move(serialized_func),
			param, mod_origin);
}

void ServerScripting::InitializeModApi(lua_State *L, int top)
{
	// Register reference classes (userdata)
	InvRef::Register(L);
	ItemStackMetaRef::Register(L);
	LuaAreaStore::Register(L);
	LuaItemStack::Register(L);
	LuaValueNoise::Register(L);
	LuaValueNoiseMap::Register(L);
	LuaPseudoRandom::Register(L);
	LuaPcgRandom::Register(L);
	LuaRaycast::Register(L);
	LuaSecureRandom::Register(L);
	LuaVoxelManip::Register(L);
	NodeMetaRef::Register(L);
	NodeTimerRef::Register(L);
	ObjectRef::Register(L);
	PlayerMetaRef::Register(L);
	LuaSettings::Register(L);
	StorageRef::Register(L);
	ModChannelRef::Register(L);

	// Initialize mod api modules
	ModApiAuth::Initialize(L, top);
	ModApiCraft::Initialize(L, top);
	ModApiEnv::Initialize(L, top);
	ModApiInventory::Initialize(L, top);
	ModApiItem::Initialize(L, top);
	ModApiMapgen::Initialize(L, top);
	ModApiParticles::Initialize(L, top);
	ModApiRollback::Initialize(L, top);
	ModApiServer::Initialize(L, top);
	ModApiUtil::Initialize(L, top);
	ModApiHttp::Initialize(L, top);
	ModApiStorage::Initialize(L, top);
	ModApiChannels::Initialize(L, top);
	ModApiIPC::Initialize(L, top);
}

void ServerScripting::InitializeAsync(lua_State *L, int top)
{
	// classes
	ItemStackMetaRef::Register(L);
	LuaAreaStore::Register(L);
	LuaItemStack::Register(L);
	LuaValueNoise::Register(L);
	LuaValueNoiseMap::Register(L);
	LuaPseudoRandom::Register(L);
	LuaPcgRandom::Register(L);
	LuaSecureRandom::Register(L);
	LuaVoxelManip::Register(L);
	LuaSettings::Register(L);

	// globals data
	auto *data = ModApiBase::getServer(L)->m_lua_globals_data.get();
	assert(data);
	script_unpack(L, data);
	lua_setfield(L, top, "transferred_globals");
}
