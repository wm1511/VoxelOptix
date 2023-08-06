#pragma once
#include <optix_types.h>
#include <vector_types.h>

struct LaunchParams
{
	int width{}, height{};
    float time{};
	float4* frame_pointer = nullptr;

    struct
    {
	    float3 position{}, starting_point{}, horizontal_map{}, vertical_map{};
    } camera;

    OptixTraversableHandle traversable{};
};

struct RayGenData
{
    void* data;
};


struct MissData
{
    void* data;
};


struct HitGroupData
{
    void* data;
};