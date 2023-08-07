#pragma once
#include "Chunk.hpp"

class World
{
public:
	World();

	void HandleReconstruction(float3 camera_position);
	std::vector<Chunk>& GetChunks() { return chunks_; }
	[[nodiscard]] bool NeedsReconstruction() const { return needs_reconstruction_; }

private:
	void Expand(int3 camera_chunk);
	void Shrink(int3 camera_chunk);
	bool CheckChunk(int3 coords);

	std::vector<Chunk> chunks_{};
	bool needs_reconstruction_{};
	int generation_distance_ = 4;

};
