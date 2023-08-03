#pragma once

__forceinline__ __host__ __device__ unsigned rotl(const unsigned x, const int k)
{
	return (x << k) | (x >> (32 - k));
}

__forceinline__ __host__ __device__ float pcg(unsigned* random_state)
{
	const unsigned state = *random_state;
	*random_state = *random_state * 747796405u + 2891336453u;
	const unsigned word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (float)(((word >> 22u) ^ word) >> 8) * (1.0f / (1ul << 24));
}

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