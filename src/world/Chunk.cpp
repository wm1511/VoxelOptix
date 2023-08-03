#include "Chunk.hpp"
#include "../misc/Noise.hpp"

Chunk::Chunk(const float3 coords) :
	coords_(coords)
{
	/*for (unsigned char x = 0; x < size_; x++)
	{
		for (unsigned char z = 0; z < size_; z++)
		{
			const float2 p = make_float2(static_cast<float>(x) + pcg(&random_state), static_cast<float>(z) + pcg(&random_state));

			const float factor = fabs(Fbm2D(p * 0.05f, 2) * 20.0f);
			const auto height = static_cast<unsigned>(clamp(factor, 1.0f, static_cast<float>(size_)));

			voxels_[x][height][z] = 1;
		}
	}*/

	for (unsigned char x = 0; x < size_; x++)
	{
		for (unsigned char y = 0; y < size_; y++)
		{
			for (unsigned char z = 0; z < size_; z++)
			{
				const float3 p = GetPosition() + make_float3(x, y, z);

				const float filling_factor = Fbm3D(p * 0.1f, 3) + 0.2f;

				voxels_[x][y][z] = filling_factor > 0.0f ? 1 : 0;
			}
		}
	}
}