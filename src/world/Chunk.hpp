#pragma once

class Chunk
{
public:
	explicit Chunk(int3 coords);

	[[nodiscard]] unsigned char GetVoxel(const unsigned char x, const unsigned char y, const unsigned char z) const { return voxels_[x][y][z]; }
	[[nodiscard]] int3 GetCoords() const { return coords_; }
	[[nodiscard]] float3 GetPosition() const { return make_float3(size_ * coords_); }

	static constexpr unsigned char size_ = 16;

private:
	unsigned char voxels_[size_][size_][size_]{};
	int3 coords_{};
};