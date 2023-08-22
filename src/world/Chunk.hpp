#pragma once
#include <optix_types.h>

class Chunk
{
public:
	Chunk() = default;
	explicit Chunk(int3 coords);

	static constexpr unsigned char size_ = 16;

	[[nodiscard]] unsigned char GetVoxel(const unsigned char x, const unsigned char y, const unsigned char z) const { return voxels_[x + y * size_ + z * size_ * size_]; }
	[[nodiscard]] float3 GetPosition() const { return make_float3(size_ * coords_); }
	[[nodiscard]] decltype(auto) GetVoxels() const { return voxels_; }
	[[nodiscard]] OptixTraversableHandle& GetIasHandle() { return ias_handle_; }
	[[nodiscard]] void*& GetIasBuffer() { return ias_buffer_; }

private:
	std::array<unsigned char, static_cast<size_t>(size_* size_* size_)> voxels_{};
	int3 coords_{};
	void* ias_buffer_ = nullptr;
	OptixTraversableHandle ias_handle_{};
};
