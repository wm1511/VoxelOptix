#pragma once
#include <optix_types.h>

class Chunk
{
public:
	Chunk() = default;
	explicit Chunk(int3 coords);

	static constexpr unsigned char size_ = 16;

	[[nodiscard]] unsigned char GetVoxel(const unsigned char x, const unsigned char y, const unsigned char z) const { return voxels_[x + y * size_ + z * size_ * size_]; }
	[[nodiscard]] int3 GetCoords() const { return coords_; }
	[[nodiscard]] float3 GetPosition() const { return make_float3(size_ * coords_); }
	[[nodiscard]] decltype(auto) GetVoxels() const { return voxels_; }
	[[nodiscard]] float GetDistance() const { return distance_; }
	[[nodiscard]] OptixTraversableHandle& GetIasHandle() { return ias_handle_; }
	[[nodiscard]] void*& GetIasBuffer() { return ias_buffer_; }

	void UpdateDistance(const float3 camera_position) { distance_ = length(fabs(camera_position - make_float3(coords_ * size_))); }

private:
	std::array<unsigned char, static_cast<size_t>(size_* size_* size_)> voxels_{};
	int3 coords_{};
	float distance_{};
	void* ias_buffer_ = nullptr;
	OptixTraversableHandle ias_handle_{};
};
