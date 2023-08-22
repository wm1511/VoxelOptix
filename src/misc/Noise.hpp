#pragma once
// Noise generation based on Inigo Quilez articles

__forceinline__ __host__ __device__ float Hash(const float n)
{
	return fracf(sinf(n) * 753.5453123f);
}

__forceinline__ __host__ __device__ float Noise3D(const float3 x)
{
	const float3 p = floorf(x);
	const float3 w = fracf(x);
	const float3 u = w * w * w * (w * (w * 6.0f - 15.0f) + 10.0f);

	const float n = p.x + 317.0f * p.y + 157.0f * p.z;

	const float a = Hash(n + 0.0f);
	const float b = Hash(n + 1.0f);
	const float c = Hash(n + 317.0f);
	const float d = Hash(n + 318.0f);
	const float e = Hash(n + 157.0f);
	const float f = Hash(n + 158.0f);
	const float g = Hash(n + 474.0f);
	const float h = Hash(n + 475.0f);

	const float k0 = a;
	const float k1 = b - a;
	const float k2 = c - a;
	const float k3 = e - a;
	const float k4 = a - b - c + d;
	const float k5 = a - c - e + g;
	const float k6 = a - b - e + f;
	const float k7 = -a + b + c - d + e - f - g + h;

	return -1.0f + 2.0f * (k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z);
}

__forceinline__ __host__ __device__ float Noise2D(const float2 x)
{
	const float2 p = floorf(x);
	const float2 w = fracf(x);
	const float2 u = w * w * w * (w * (w * 6.0f - 15.0f) + 10.0f);

	const float n = p.x + 317.0f * p.y;

	const float a = Hash(n + 0.0f);  
	const float b = Hash(n + 1.0f);  
	const float c = Hash(n + 317.0f);
	const float d = Hash(n + 318.0f);

	return -1.0f + 2.0f * (a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y);
}

__forceinline__ __host__ __device__ float Fbm3D(float3 in, const unsigned octaves)
{
	float out = 0.0f;
	float amplitude = 0.5f;

	for (unsigned i = 0; i < octaves; i++)
	{
		out += amplitude * Noise3D(in);
		in *= 2.0f;
		amplitude *= 0.5f;
	}

	return out;
}

__forceinline__ __host__ __device__ float Fbm2D(float2 in, const unsigned octaves)
{
	float out = 0.0f;
	float amplitude = 0.5f;

	for (unsigned i = 0; i < octaves; i++)
	{
		out += amplitude * Noise2D(in);
		in *= 2.0f;
		amplitude *= 0.5f;
	}

	return out;
}