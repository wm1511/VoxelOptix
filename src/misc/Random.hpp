#pragma once

__forceinline__ __host__ __device__ unsigned rotl(const unsigned x, const int k)
{
	return (x << k) | (x >> (32 - k));
}

// pcg_rxs_m_xs
__forceinline__ __host__ __device__ float pcg(unsigned* random_state)
{
	const unsigned state = *random_state;
	*random_state = *random_state * 747796405u + 2891336453u;
	const unsigned word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return static_cast<float>(((word >> 22u) ^ word) >> 8) * (1.0f / (1ul << 24));
}

// xoshiro128+
__forceinline__ __host__ __device__ unsigned xoshiro(uint4* random_state)
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

__forceinline__ __host__ __device__ float2 disk_random(unsigned* random_state)
{
	float2 v;
	do
	{
		v = 2.0f * make_float2(pcg(random_state), pcg(random_state)) - make_float2(1.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}

__forceinline__ __host__ __device__ float3 sphere_random(unsigned* random_state)
{
	float3 v;
	do
	{
		v = make_float3(pcg(random_state), pcg(random_state), pcg(random_state)) - make_float3(1.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}