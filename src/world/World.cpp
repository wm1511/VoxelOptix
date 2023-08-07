#include "World.hpp"

World::World()
{
	for (int x = -generation_distance_; x < generation_distance_; x++)
	{
		for (int y = 0; y < generation_distance_ / 2; y++)
		{
			for (int z = -generation_distance_; z < generation_distance_; z++)
			{
				chunks_.emplace_back(make_int3(x, y, z));
			}
		}
	}
}

void World::HandleReconstruction(const float3 camera_position)
{
	needs_reconstruction_ = false;

	const int3 camera_chunk = make_int3(camera_position / Chunk::size_);

	Expand(camera_chunk);
	Shrink(camera_chunk);
}

void World::Expand(const int3 camera_chunk)
{
	for (int x = camera_chunk.x - generation_distance_; x < camera_chunk.x + generation_distance_; x++)
	{
		for (int y = camera_chunk.y - generation_distance_ / 2; y < generation_distance_ / 2; y++)
		{
			for (int z = camera_chunk.z - generation_distance_; z < camera_chunk.z + generation_distance_; z++)
			{
				needs_reconstruction_ |= CheckChunk(make_int3(x, y, z));
			}
		}
	}
}

void World::Shrink(int3 camera_chunk)
{
	needs_reconstruction_ |= static_cast<bool>(std::erase_if(chunks_, [this, &camera_chunk](const Chunk& c)
		{
			const int3 distance = c.GetCoords() - camera_chunk;
			return abs(distance.x) > 4 * generation_distance_ ||
				distance.y > 4 * generation_distance_ ||
				abs(distance.z) > 4 * generation_distance_;
		}));
}

bool World::CheckChunk(int3 coords)
{
	for (auto& chunk : chunks_)
	{
		const int3 chunk_coords = chunk.GetCoords();
		if (chunk_coords.x == coords.x && chunk_coords.y == coords.y && chunk_coords.z == coords.z)
			return false;
	}

	chunks_.emplace_back(coords);
	return true;
}
