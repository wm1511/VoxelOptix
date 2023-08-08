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

	__forceinline__ __device__ void trace(float3& origin, float3& direction, float3& color, unsigned& random_state, unsigned& depth)
	{
		unsigned u0 = __float_as_uint(color.x);
		unsigned u1 = __float_as_uint(color.y);
		unsigned u2 = __float_as_uint(color.z);
		unsigned u3 = random_state;
		unsigned u4 = depth;
		unsigned u5 = __float_as_uint(origin.x);
		unsigned u6 = __float_as_uint(origin.y);
		unsigned u7 = __float_as_uint(origin.z);
		unsigned u8 = __float_as_uint(direction.x);
		unsigned u9 = __float_as_uint(direction.y);
		unsigned u10 = __float_as_uint(direction.z);

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
			u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10);

		origin = make_float3(__uint_as_float(u5), __uint_as_float(u6), __uint_as_float(u7));
		direction = make_float3(__uint_as_float(u8), __uint_as_float(u9), __uint_as_float(u10));
		color = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
		random_state = u3;
		depth = u4;
	}
}

extern "C" __global__ void __closesthit__triangle()
{
	const unsigned index = optixGetPrimitiveIndex();
	unsigned random_state = optixGetPayload_3();

	float3 color = make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()));

	const float3 reflected_direction = kNormals[index / 2] + sphere_random(&random_state);
	const float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

	color *= make_float3(0.5f);

	optixSetPayload_0(__float_as_uint(color.x));
	optixSetPayload_1(__float_as_uint(color.y));
	optixSetPayload_2(__float_as_uint(color.z));
	optixSetPayload_3(random_state);
	optixSetPayload_4(optixGetPayload_4() - 1);
	optixSetPayload_5(__float_as_uint(hit_point.x));
	optixSetPayload_6(__float_as_uint(hit_point.y));
	optixSetPayload_7(__float_as_uint(hit_point.z));
	optixSetPayload_8(__float_as_uint(reflected_direction.x));
	optixSetPayload_9(__float_as_uint(reflected_direction.y));
	optixSetPayload_10(__float_as_uint(reflected_direction.z));
}

extern "C" __global__ void __miss__sky()
{
	const float3 origin = optixGetWorldRayOrigin();
	const float3 direction = optixGetWorldRayDirection();

	float3 color = make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()));

	color *= RenderSky(origin, direction);
	const float4 cloud_color = RenderClouds(origin, direction);
	color = color * (1.0f - cloud_color.w) + make_float3(cloud_color);

	optixSetPayload_0(__float_as_uint(color.x));
	optixSetPayload_1(__float_as_uint(color.y));
	optixSetPayload_2(__float_as_uint(color.z));
	optixSetPayload_4(0);
}

extern "C" __global__ void __raygen__render()
{
	const uint3 index = optixGetLaunchIndex();
	const unsigned pixel_index = index.x + index.y * launch_params.width;
	unsigned depth_remaining = launch_params.max_depth;

	// Using constant random seed to make noise stable
	unsigned random_state = pixel_index;

	const float u = (static_cast<float>(index.x) + pcg(&random_state)) / static_cast<float>(launch_params.width);
	const float v = (static_cast<float>(index.y) + pcg(&random_state)) / static_cast<float>(launch_params.height);

	float3 pixel_color = make_float3(1.0f);

	float3 origin = launch_params.camera.position;
	float3 direction = launch_params.camera.starting_point + u * launch_params.camera.horizontal_map + v * launch_params.camera.vertical_map - origin;

	do
	{
		trace(origin, direction, pixel_color, random_state, depth_remaining);
	} while (depth_remaining > 0);

	launch_params.frame_pointer[pixel_index] = make_float4(powf(pixel_color, 1.0f / 2.2f), 1.0f);
}