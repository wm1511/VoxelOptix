#include "LaunchParams.hpp"
#include "../external/helper_math.h"
#include "../misc/Random.hpp"
#include "../misc/Noise.hpp"

#include <vector_types.h>
#include <optix_device.h>

__constant__ LaunchParams launch_params;

// Normals of voxel stored for fast access
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
	__forceinline__ __device__ float3 Fog(const float3 color, const float t)
	{
		const float3 intensity = exp2f(-t * 0.005f * make_float3(0.1f, 0.15f, 0.4f));
		// Fog color based on wall paint named "Transparent Fog"
		return color * intensity + (1.0f - intensity) * make_float3(0.776f, 0.843f, 0.847f);
	}

	__forceinline__ __device__ float3 RenderSky(const float3& origin, const float3& direction)
	{
		// Sky color gradient
		float3 color = make_float3(0.4f, 0.6f, 1.1f) - direction.y * 0.4f;

		// Cirrocumulus has its base at 6-8km
		const float t = (7000.0f - origin.y) / direction.y;
		if (t > 0.0f)
		{
			const float3 uv = origin + t * direction;
			const float noise = Fbm2D(make_float2(uv.x, uv.z) * 5e-4f + make_float2(0.0f, -launch_params.time * 0.1f), 2);
			const float brightness = 0.1f * smoothstep(-0.2f, 0.6f, noise);
			color = lerp(color, make_float3(1.0f), brightness / (t * 2e-4f));
		}

		return color;
	}

	__forceinline__ __device__ float4 RenderClouds(const float3& origin, const float3& direction)
	{
		constexpr int steps = 8;
		constexpr float cloud_thickness = 20.0f;
		constexpr float march_step = cloud_thickness / static_cast<float>(steps);

		const float3 projection = direction / direction.y;
		const float cutoff = dot(direction, make_float3(0.0f, 1.0f, 0.0f));

		// Prevent clouds from rendering on bottom hemisphere
		if (cutoff < 0.0f)
			return make_float4(0.0f);

		// Cumulus has its base at 500-1500m 
		const float3 cloud_origin = origin + projection * 500.0f;
		float3 position = cloud_origin;

		float transmittance = 1.0f;
		float3 color = make_float3(0.0f, 0.0f, 0.0f);
		float alpha = 0.0f;

		for (int i = 0; i < steps; i++)
		{
			float density = Fbm3D(position * 0.002f + make_float3(0.0f, 0.0f, -launch_params.time * 0.1f), 2);
			// Density decreases further away from the camera
			density *= smoothstep(0.2f, 0.3f, density / (length(origin - position) * 5e-4f));

			const float delta = expf(-density * march_step);
			transmittance *= delta;
			// Generating brighter colors in higher part of cloud
			color += transmittance * expf((position.y - cloud_origin.y) / cloud_thickness) * 0.5f * density * march_step;
			alpha += (1.0f - delta) * (1.0f - alpha);

			position += projection * march_step;

			if (alpha > 0.999f)
				break;
		}

		return make_float4(color, alpha * smoothstep(0.0f, 0.3f, cutoff));
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

	float3 color = make_float3(0.5f, 0.5f, 0.5f) * shading;
	// Applying fog to objects further away from the camera
	color = Fog(color, optixGetRayTmax());

	optixSetPayload_0(__float_as_uint(color.x));
	optixSetPayload_1(__float_as_uint(color.y));
	optixSetPayload_2(__float_as_uint(color.z));
}

extern "C" __global__ void __miss__sky()
{
	const float3 origin = optixGetWorldRayOrigin();
	const float3 direction = optixGetWorldRayDirection();

	float3 color = RenderSky(origin, direction);
	const float4 cloud_color = RenderClouds(origin, direction);
	color = color * (1.0f - cloud_color.w) + make_float3(cloud_color);

	optixSetPayload_0(__float_as_uint(color.x));
	optixSetPayload_1(__float_as_uint(color.y));
	optixSetPayload_2(__float_as_uint(color.z));
}

extern "C" __global__ void __raygen__render()
{
	const uint3 index = optixGetLaunchIndex();
	const unsigned pixel_index = index.x + index.y * launch_params.width;

	// Using frame memory as seed for RNG
	unsigned random_state = xoshiro((uint4*)launch_params.frame_pointer + pixel_index);

	const float u = (static_cast<float>(index.x) + pcg(&random_state)) / static_cast<float>(launch_params.width);
	const float v = (static_cast<float>(index.y) + pcg(&random_state)) / static_cast<float>(launch_params.height);

	float3 pixel_color = make_float3(1.0f);

	const float3 origin = launch_params.camera.position;
	const float3 direction = launch_params.camera.starting_point + u * launch_params.camera.horizontal_map + v * launch_params.camera.vertical_map - origin;

	trace(origin, direction, pixel_color);

	launch_params.frame_pointer[pixel_index] = make_float4(powf(pixel_color, 1.0f / 2.2f), 1.0f);
}