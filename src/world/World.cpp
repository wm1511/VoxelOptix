#include "World.hpp"
#include "../misc/Exceptions.hpp"

World::World()
{
	for (int x = -generation_distance_; x < generation_distance_; x++)
	{
		for (int y = -generation_distance_; y < 0; y++)
		{
			for (int z = -generation_distance_; z < generation_distance_; z++)
			{
				const int3 coords = make_int3(x, y, z);

				chunks_.emplace(coords, coords);
			}
		}
	}
}

void World::CheckForUpdate(const float3 camera_position)
{
	const auto camera_chunk = make_int3(camera_position / Chunk::size_);

	if (camera_chunk.x == camera_chunk_.x && camera_chunk.y == camera_chunk_.y && camera_chunk.z == camera_chunk_.z)
		return;

	camera_chunk_ = camera_chunk;
	needs_update_ = true;
}

void World::HandleUpdate()
{
	Expand();
	Shrink();
}

void World::Expand()
{
	for (int x = camera_chunk_.x - generation_distance_; x < camera_chunk_.x + generation_distance_; x++)
	{
		for (int y = camera_chunk_.y - generation_distance_; y < 0; y++)
		{
			for (int z = camera_chunk_.z - generation_distance_; z < camera_chunk_.z + generation_distance_; z++)
			{
				const int3 coords = make_int3(x, y, z);
				chunks_.try_emplace(coords, coords);
			}
		}
	}
}

void World::Shrink()
{
	for (auto& [coords, chunk] : chunks_)
	{
		if (!(coords.x >= camera_chunk_.x - generation_distance_ && coords.x < camera_chunk_.x + generation_distance_ &&
			coords.y >= camera_chunk_.y - generation_distance_ && coords.y < 0 &&
			coords.z >= camera_chunk_.z - generation_distance_ && coords.z < camera_chunk_.z + generation_distance_))
		{
			CCE(cudaFree(chunk.GetIasBuffer()));
			chunk.GetIasHandle() = 0;
			chunks_.erase(coords);
		}
	}
}
