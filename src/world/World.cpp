#include "World.hpp"

World::World()
{
	chunks_.emplace_back(make_float3(0.0f));
	chunks_.emplace_back(make_float3(1.0f));
	chunks_.emplace_back(make_float3(0.0f, 0.0f, -1.0f));
}
