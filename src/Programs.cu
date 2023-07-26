#include "LaunchParams.hpp"
#include "helper_math.h"

#include <vector_types.h>
#include <optix_device.h>

__constant__ LaunchParams launch_params;

__constant__ constexpr float kPi = 3.141593f;
__constant__ constexpr float k2Pi = 6.283185f;
__constant__ constexpr float kHalfPi = 1.570796f;
__constant__ constexpr float kInvPi = 0.318309f;
__constant__ constexpr float kInv2Pi = 0.159154f;
__constant__ constexpr float kTMin = 0.001f;
__constant__ constexpr float kFMax = 3.402823466e+38F;
__constant__ constexpr float kFMin = 1.175494351e-38F;
__constant__ constexpr float kFEps = 1.192092896e-07F;

extern unsigned __float_as_uint(float x);
extern float __uint_as_float(unsigned x);

static __forceinline__ __device__ float pcg(unsigned* random_state)
{
	unsigned state = *random_state;
	*random_state = *random_state * 747796405u + 2891336453u;
	unsigned word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (float)(((word >> 22u) ^ word) >> 8) * (1.0f / (UINT32_C(1) << 24));
}

static __forceinline__ __device__ float2 disk_random(unsigned* random_state)
{
	float2 v;
	do
	{
		v = 2.0f * make_float2(pcg(random_state), pcg(random_state)) - make_float2(1.0f, 1.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}

static __forceinline__ __device__ void cast_ray(float3& origin, float3& direction, unsigned* random_state, const float screen_x, const float screen_y, const CameraInfo& camera_info)
{
	const float2 random_on_lens = camera_info.lens_radius * disk_random(random_state);
	const float3 ray_offset = camera_info.u * random_on_lens.x + camera_info.v * random_on_lens.y;
	origin = camera_info.position + ray_offset;
	direction = camera_info.starting_point + screen_x * camera_info.horizontal_map + screen_y * camera_info.vertical_map - camera_info.position - ray_offset;
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
		kTMin,
		kFMax,
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
	optixSetPayload_0(__float_as_uint(0.8f));
	optixSetPayload_1(__float_as_uint(0.8f));
	optixSetPayload_2(__float_as_uint(1.0f));
}

extern "C" __global__ void __raygen__render()
{
	const uint3 index = optixGetLaunchIndex();
	const unsigned pixel_index = index.x + index.y * launch_params.width;

	unsigned random_state = pixel_index;
	const float u = (static_cast<float>(index.x) + pcg(&random_state)) / static_cast<float>(launch_params.width);
	const float v = (static_cast<float>(index.y) + pcg(&random_state)) / static_cast<float>(launch_params.height);

	float3 pixel_color = make_float3(1.0f);
	float3 origin;
	float3 direction;

	cast_ray(origin, direction, &random_state, u, v, launch_params.camera_info);

	trace(origin, direction, pixel_color);

	launch_params.frame_buffer[pixel_index] = make_float4(pixel_color, 1.0f);
}