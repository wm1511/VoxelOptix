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

	void CheckForUpdate(float3 camera_position);
	void HandleUpdate();

	void ResetUpdateFlag() { needs_update_ = false; }
	void SetUpdateFlag() { needs_update_ = true; }
	std::unordered_map<int3, Chunk>& GetChunks() { return chunks_; }
	Chunk& GetChunk(const int3 coords) { return chunks_[coords]; }
	[[nodiscard]] int& GetGenerationDistance() { return generation_distance_; }
	[[nodiscard]] int3 GetCameraChunk() const { return camera_chunk_; }
	[[nodiscard]] bool NeedsUpdate() const { return needs_update_; }

private:
	void Expand();
	void Shrink();

	std::unordered_map<int3, Chunk> chunks_{};
	int generation_distance_{8};
	int3 camera_chunk_{};
	bool needs_update_{};
};
