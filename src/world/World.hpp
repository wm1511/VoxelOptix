#pragma once
#include "Chunk.hpp"

class World
{
public:
	World();

	std::vector<Chunk>& GetChunks() { return chunks_; }

private:
	std::vector<Chunk> chunks_{};

};
