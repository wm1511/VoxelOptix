#include "LaunchParams.hpp"
#include "../external/helper_math.h"

#include <vector_types.h>
#include <optix_device.h>

__constant__ LaunchParams launch_params;
__constant__ constexpr float3 kSunDir = { -0.624695f,0.468521f,-0.624695f };
__constant__ constexpr unsigned kNoiseOctaves = 4;
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

static __forceinline__ __device__ unsigned rotl(const unsigned x, const int k)
{
	return (x << k) | (x >> (32 - k));
}

static __forceinline__ __device__ float pcg(unsigned* random_state)
{
	const unsigned state = *random_state;
	*random_state = *random_state * 747796405u + 2891336453u;
	const unsigned word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
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

static __forceinline__ __device__ float hash1(float2 p)
{
	p = 50.0f * fracf(p * 0.3183099f);
	return fracf(p.x * p.y * (p.x + p.y));
}

static __forceinline__ __device__ float hash1(const float n)
{
	return fracf(n * 17.0f * fracf(n * 0.3183099f));
}

static __forceinline__ __device__ float4 noised(const float3 x)
{
	const float3 p = floorf(x);
	const float3 w = fracf(x);
	const float3 u = w * w * w * (w * (w * 6.0f - 15.0f) + 10.0f);
	const float3 du = 30.0f * w * w * (w * (w - 2.0f) + 1.0f);

	const float n = p.x + 317.0f * p.y + 157.0f * p.z;

	const float a = hash1(n + 0.0f);
	const float b = hash1(n + 1.0f);
	const float c = hash1(n + 317.0f);
	const float d = hash1(n + 318.0f);
	const float e = hash1(n + 157.0f);
	const float f = hash1(n + 158.0f);
	const float g = hash1(n + 474.0f);
	const float h = hash1(n + 475.0f);

	const float k0 = a;
	const float k1 = b - a;
	const float k2 = c - a;
	const float k3 = e - a;
	const float k4 = a - b - c + d;
	const float k5 = a - c - e + g;
	const float k6 = a - b - e + f;
	const float k7 = -a + b + c - d + e - f - g + h;

	const float3 l = 2.0f * du * make_float3(k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
			k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
			k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);

	return make_float4(
		-1.0f + 2.0f * (k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.
			x * u.y * u.z), l.x, l.y, l.z);
}

static __forceinline__ __device__ float noise(const float2 x)
{
	const float2 p = floorf(x);
	const float2 w = fracf(x);
	const float2 u = w * w * w * (w * (w * 6.0f - 15.0f) + 10.0f);

	const float a = hash1(p + make_float2(0.0f, 0.0f));
	const float b = hash1(p + make_float2(1.0f, 0.0f));
	const float c = hash1(p + make_float2(0.0f, 1.0f));
	const float d = hash1(p + make_float2(1.0f, 1.0f));

	return -1.0f + 2.0f * (a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y);
}

static __forceinline__ __device__ float4 fbmd(float3 x)
{
	float a = 0.0f;
	float b = 0.5f;
	float3 d = make_float3(0.0f);
	for (unsigned i = 0; i < kNoiseOctaves; i++)
	{
		const float4 n = noised(x);
		a += b * n.x;	
		if (i < 4)
			d += b * make_float3(n.y, n.z, n.w);
		b *= 0.65f;
		x *= 2.0f;
	}
	return make_float4(a, d.x, d.y, d.z);
}

static __forceinline__ __device__ float fbm(float2 x)
{
	float a = 0.0f;
	float b = 0.5f;
	for (unsigned i = 0; i < kNoiseOctaves; i++)
	{
		const float n = noise(x);
		a += b * n;
		b *= 0.55f;
		x *= 1.9f;
	}

	return a;
}

static __forceinline__ __device__ float3 RenderSky(const float3& origin, const float3& direction)
{
	float3 color = make_float3(0.42f, 0.62f, 1.1f) - direction.y * 0.4f;

	const float t = (2500.0f - origin.y) / direction.y;
	if (t > 0.0f)
	{
		const float3 uv = origin + t * direction;
		const float cl = fbm(make_float2(uv.x, uv.z) * 0.00104f);
		const float dl = smoothstep(-0.2f, 0.6f, cl);
		color = lerp(color, make_float3(1.0f), 0.12f * dl);
	}

	return color;
}

static __forceinline__ __device__ float3 Fog(const float3 col, const float t)
{
	const float3 ext = exp2f(-t * 0.00025f * make_float3(1.0f, 1.5f, 4.0f));
	return col * ext + (1.0f - ext) * float3(0.55f, 0.55f, 0.58f);
}

static __forceinline__ __device__ float4 CloudsFbm(const float3 pos)
{
	return fbmd(pos * 0.0015f + make_float3(2.0f, 1.1f, 1.0f) + 0.07f * make_float3(launch_params.time, 0.5f * launch_params.time, -0.15f * launch_params.time));
}

static __forceinline__ __device__ float4 CloudsMap(const float3 pos, float& nnd)
{
	float d = fabsf(pos.y - 900.0f) - 40.0f;
	const float3 gra = make_float3(0.0f, pos.y - 900.0f > 0.0f ? 1.0f : -1.0f, 0.0f);

	const float4 n = CloudsFbm(pos);
	d += 400.0f * n.x * (0.7f + 0.3f * gra.y);

	if (d > 0.0f)
		return make_float4(-d, 0.0f, 0.0f, 0.0f);

	nnd = -d;
	d = fminf(-d / 100.0f, 0.25f);

	return make_float4(d, gra.x, gra.y, gra.z);
}

static __forceinline__ __device__ float4 RenderClouds(const float3& origin, const float3& direction, float t_min, float t_max)
{
	float4 sum = make_float4(0.0f);

	const float tl = (600.0f - origin.y) / direction.y;
	const float th = (1200.0f - origin.y) / direction.y;

	if (tl > 0.0f)
		t_min = fmaxf(t_min, tl);
	else
		return sum;
	if (th > 0.0f)
		t_max = fminf(t_max, th);

	float t = t_min;

	float last_t = -1.0;

	for (int i = 0; i < 128; i++)
	{
		const float3 pos = origin + t * direction;
		float nnd;
		const float4 den_gra = CloudsMap(pos, nnd);
		const float den = den_gra.x;
		float dt = fmaxf(0.2f, 0.011f * t);

		if (den > 0.001f)
		{
			float kk;
			CloudsMap(pos + kSunDir * 70.0f, kk);
			float sha = 1.0f - smoothstep(-200.0f, 200.0f, kk);
			sha *= 1.5f;

			const float3 nor = normalize(make_float3(den_gra.y, den_gra.z, den_gra.w));
			const float dif = clamp(0.4f + 0.6f * dot(nor, kSunDir), 0.0f, 1.0f) * sha;
			const float occ = 0.2f + 0.7f * fmaxf(1.0f - kk / 200.0f, 0.0f) + 0.1f * (1.0f - den);

			float3 lin = make_float3(0.0f);
			lin += make_float3(0.70f, 0.80f, 1.00f) * 1.0f * (0.5f + 0.5f * nor.y) * occ;
			lin += make_float3(0.10f, 0.40f, 0.20f) * 1.0f * (0.5f - 0.5f * nor.y) * occ;
			lin += make_float3(1.00f, 0.95f, 0.85f) * 3.0f * dif * occ + 0.1f;

			float3 col = make_float3(0.8f, 0.8f, 0.8f) * 0.45f;
			col *= lin;
			col = Fog(col, t);

			const float alp = clamp(den * 0.5f * 0.125f * dt, 0.0f, 1.0f);
			col *= alp;
			sum = sum + make_float4(col, alp) * (1.0f - sum.w);

			if (last_t < 0.0f)
				last_t = t;
		}
		else
		{
			dt = fabsf(den) + 0.2f;
		}
		t += dt;

		if (sum.w > 0.995f || t > t_max)
			break;
	}

	return clamp(sum, 0.0f, 1.0f);
}

static __forceinline__ __device__ void trace(const float3& origin, const float3& direction, float3& color)
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

	float3 color = RenderSky(origin, direction);
	const float4 result = RenderClouds(origin, direction, 0.0f, 5000.0f);
	color = color * (1.0f - result.w) + make_float3(result.x, result.y, result.z);

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