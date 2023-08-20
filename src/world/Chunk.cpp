#include "Chunk.hpp"
#include "../misc/Noise.hpp"

Chunk::Chunk(const int3 coords) :
	coords_(coords),
	distance_(length(make_float3(coords * size_)))
{
	for (unsigned char x = 0; x < size_; x++)
	{
		for (unsigned char y = 0; y < size_; y++)
		{
			for (unsigned char z = 0; z < size_; z++)
			{
				const unsigned index = x + y * size_ + z * size_ * size_;
				const float3 p = GetPosition() + make_float3(x, y, z);

				const float filling_factor = Fbm3D(p * 0.1f, 3) + 0.2f;

				if (coords.y == -1)
				{
					const float height_factor = smoothstep(-1.0f, 2.0f, Fbm2D(make_float2(p.x, p.z) * 0.02f, 3));

					if (y <= static_cast<unsigned char>(height_factor * static_cast<float>(size_)))
						voxels_[index] = filling_factor > 0.0f ? 1 : 0;
				}
				else
					voxels_[index] = filling_factor > 0.0f ? 1 : 0;
			}
		}
	}
}