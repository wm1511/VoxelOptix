#include "LaunchParams.hpp"
#include "../external/helper_math.h"

#include <vector_types.h>
#include <optix_device.h>

__constant__ LaunchParams launch_params;

extern unsigned __float_as_uint(float x);
extern float __uint_as_float(unsigned x);

static __forceinline__ __device__ unsigned rotl(const unsigned x, const int k)
{
	return (x << k) | (x >> (32 - k));
}

static __forceinline__ __device__ float pcg(unsigned* random_state)
{
	unsigned state = *random_state;
	*random_state = *random_state * 747796405u + 2891336453u;
	unsigned word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (float)(((word >> 22u) ^ word) >> 8) * (1.0f / (UINT32_C(1) << 24));
}

static __forceinline__ __device__ unsigned xoshiro(uint4* random_state)
{
	const unsigned result = random_state->x + random_state->w;
	const unsigned t = random_state->y << 9;

	random_state->z ^= random_state->x;
	random_state->w ^= random_state->y;
	random_state->y ^= random_state->z;
	random_state->x ^= random_state->w;
	random_state->z ^= t;
	random_state->w = rotl(random_state->w, 11);

	return result;
}

static __forceinline__ __device__ void trace(float3& origin, float3& direction, float3& color)
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

extern "C" __global__ void __closesthit__triangle()
{
	optixSetPayload_0(__float_as_uint(1.0f));
	optixSetPayload_1(__float_as_uint(0.0f));
	optixSetPayload_2(__float_as_uint(0.0f));
}

extern "C" __global__ void __miss__sky()
{
	const float3 color = lerp(make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), dot(make_float3(0.0f, 1.0f, 0.0f), optixGetWorldRayDirection()));

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

	float3 origin = launch_params.camera.position;
	float3 direction = launch_params.camera.starting_point + u * launch_params.camera.horizontal_map + v * launch_params.camera.vertical_map - origin;

	trace(origin, direction, pixel_color);

	launch_params.frame_buffer[pixel_index] = make_float4(pixel_color, 1.0f);
}