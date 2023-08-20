#pragma once
#include "Chunk.hpp"

template <>
struct std::hash<int3>
{
    size_t operator()(const int3& val) const noexcept
    {
        return static_cast<size_t>(val.x) ^ static_cast<size_t>(val.y) << 16 ^ static_cast<size_t>(val.z) << 32;
    }
};

template <>
struct std::equal_to<int3>
{
	bool operator()(const int3& a, const int3& b) const
	{
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}
};

class World
{
public:
	World();

	void HandleReconstruction(float3 camera_position);
	std::unordered_map<int3, Chunk>& GetChunks() { return chunks_; }
	Chunk& GetChunk(const int3 coords) { return chunks_[coords]; }
	[[nodiscard]] bool NeedsReconstruction() const { return needs_reconstruction_; }

private:
	void Expand(int3 camera_chunk);
	void Shrink(int3 camera_chunk);

	std::unordered_map<int3, Chunk> chunks_{};
	int3 last_camera_chunk_{};
	bool needs_reconstruction_{};
	int generation_distance_ = 8;
};
