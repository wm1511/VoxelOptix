#include "World.hpp"
#include "../misc/Exceptions.hpp"

World::World()
{
	for (int x = -generation_distance_; x < generation_distance_; x++)
	{
		for (int y = -generation_distance_ / 2; y < 0; y++)
		{
			for (int z = -generation_distance_; z < generation_distance_; z++)
			{
				const int3 coords = make_int3(x, y, z);

				chunks_.emplace(coords, coords);
			}
		}
	}
}

void World::HandleReconstruction(const float3 camera_position)
{
	needs_reconstruction_ = false;

	const int3 camera_chunk = make_int3(camera_position / Chunk::size_);

	if (camera_chunk.x == last_camera_chunk_.x && camera_chunk.y == last_camera_chunk_.y && camera_chunk.z == last_camera_chunk_.z)
		return;

	for (auto& chunk : chunks_ | std::views::values)
		chunk.UpdateDistance(camera_position);

	Expand(camera_chunk);
	Shrink(camera_chunk);

	last_camera_chunk_ = camera_chunk;
}

void World::Expand(const int3 camera_chunk)
{
	for (int x = camera_chunk.x - generation_distance_; x < camera_chunk.x + generation_distance_; x++)
	{
		for (int y = camera_chunk.y - generation_distance_ / 2; y < 0; y++)
		{
			for (int z = camera_chunk.z - generation_distance_; z < camera_chunk.z + generation_distance_; z++)
			{
				const int3 coords = make_int3(x, y, z);

				if (chunks_.try_emplace(coords, coords).second)
					needs_reconstruction_ = true;
			}
		}
	}
}

void World::Shrink(const int3 camera_chunk)
{
	for (auto& [coords, chunk] : chunks_)
	{
	    if (!(coords.x >= camera_chunk.x - generation_distance_ && coords.x < camera_chunk.x + generation_distance_ && 
			coords.y >= camera_chunk.y - generation_distance_ / 2 && coords.y < 0 && 
			coords.z >= camera_chunk.z - generation_distance_ && coords.z < camera_chunk.z + generation_distance_))
	    {
			CCE(cudaFree(chunk.GetIasBuffer()));
			chunks_.erase(coords);
			needs_reconstruction_ = true;	    
	    }
	}
}
