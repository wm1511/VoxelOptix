#include "LaunchParams.hpp"
#include "../external/helper_math.h"
#include "../misc/Random.hpp"
#include "../misc/Noise.hpp"

#include <vector_types.h>
#include <optix_device.h>

__constant__ LaunchParams launch_params;
__constant__ constexpr float3 kNormals[6]
{
	{0.0f, 1.0f, 0.0f},
	{0.0f, -1.0f, 0.0f},
	{-1.0f, 0.0f, 0.0f},
	{1.0f, 0.0f, 0.0f},
	{0.0f, 0.0f, -1.0f},
	{0.0f, 0.0f, 1.0f},
};

extern unsigned __float_as_uint(float x);
extern float __uint_as_float(unsigned x);

namespace
{
	__forceinline__ __device__ float3 RenderSky(const float3& origin, const float3& direction)
	{
		float3 color = make_float3(0.4f, 0.6f, 1.1f) - direction.y * 0.4f;

		// Cirrocumulus has its base at 6-8km
		const float t = (7000.0f - origin.y) / direction.y;
		if (t > 0.0f)
		{
			const float3 uv = origin + t * direction;
			const float noise = Fbm2D(make_float2(uv.x, uv.z) * 0.002f, 8);
			const float brightness = 0.3f * smoothstep(-0.2f, 0.6f, noise);
			color = lerp(color, make_float3(1.0f), brightness);
		}

		return color;
	}

	__forceinline__ __device__ void trace(const float3& origin, const float3& direction, float3& color)
	{
		unsigned u0 = __float_as_uint(color.x);
		unsigned u1 = __float_as_uint(color.y);
		unsigned u2 = __float_as_uint(color.z);

		optixTrace(
			launch_params.traversable,
			origin,
			direction,
			0.001f,
			3.402823466e+38f,
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			0,
			1,
			0,
			u0, u1, u2);

		color = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
	}
}

extern "C" __global__ void __closesthit__triangle()
{
	const unsigned index = optixGetPrimitiveIndex();
	const float shading = 0.2f + 0.8f * fabsf(dot(kNormals[index / 2], optixGetWorldRayDirection()));
	const float3 color = make_float3(0.5f, 0.5f, 0.5f) * shading;

	optixSetPayload_0(__float_as_uint(color.x));
	optixSetPayload_1(__float_as_uint(color.y));
	optixSetPayload_2(__float_as_uint(color.z));
}

extern "C" __global__ void __miss__sky()
{
	const float3 origin = optixGetWorldRayOrigin();
	const float3 direction = optixGetWorldRayDirection();

	const float3 color = RenderSky(origin, direction);

	optixSetPayload_0(__float_as_uint(color.x));
	optixSetPayload_1(__float_as_uint(color.y));
	optixSetPayload_2(__float_as_uint(color.z));
}

extern "C" __global__ void __raygen__render()
{
	const uint3 index = optixGetLaunchIndex();
	const unsigned pixel_index = index.x + index.y * launch_params.width;

	// Using frame memory as seed for RNG
	unsigned random_state = xoshiro((uint4*)launch_params.frame_buffer + pixel_index);

	const float u = (static_cast<float>(index.x) + pcg(&random_state)) / static_cast<float>(launch_params.width);
	const float v = (static_cast<float>(index.y) + pcg(&random_state)) / static_cast<float>(launch_params.height);

	float3 pixel_color = make_float3(1.0f);

	const float3 origin = launch_params.camera.position;
	const float3 direction = launch_params.camera.starting_point + u * launch_params.camera.horizontal_map + v * launch_params.camera.vertical_map - origin;

	trace(origin, direction, pixel_color);

	launch_params.frame_buffer[pixel_index] = make_float4(powf(pixel_color, 1.0f / 2.2f), 1.0f);
}