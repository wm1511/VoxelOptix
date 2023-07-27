#pragma once
#include <optix_types.h>

struct LaunchParams
{
	int width{}, height{};
	float4* frame_buffer = nullptr;

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