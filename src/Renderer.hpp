#pragma once
#include "LaunchParams.hpp"

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

class Renderer final
{
public:
	Renderer();
	~Renderer();

	Renderer(const Renderer&) = delete;
	Renderer(Renderer&&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	Renderer& operator=(Renderer&&) = delete;

	void render(float4* device_memory, int width, int height);

private:
	void init_optix();
	void create_modules();
	void create_programs();
	void create_pipeline();
	void prepare_as(const OptixBuildInput& build_input, void*& buffer, OptixTraversableHandle& handle, OptixBuildOperation operation) const;
	void prepare_gas(OptixTraversableHandle& handle, void*& buffer, OptixBuildOperation operation) const;
	void prepare_ias(std::vector<OptixTraversableHandle>& gases, OptixBuildOperation operation);
	void create_sbt();

	cudaStream_t stream_{};
	OptixDeviceContext context_ = nullptr;
	OptixModule module_ = nullptr;
	OptixPipeline pipeline_ = nullptr;
	OptixShaderBindingTable sbt_{};

	OptixPipelineCompileOptions pipeline_compile_options_{};
	OptixModuleCompileOptions module_compile_options_{};

	std::vector<OptixProgramGroup> raygen_programs_{};
	std::vector<OptixProgramGroup> miss_programs_{};
	std::vector<OptixProgramGroup> hit_programs_{};

	SbtRecord<RayGenData>* d_raygen_records_ = nullptr;
	SbtRecord<MissData>* d_miss_records_ = nullptr;
	SbtRecord<HitGroupData>* d_hit_records_ = nullptr;
	std::vector<void*> gas_buffers_{};
	std::vector<OptixTraversableHandle> gas_handles_{};
	void* ias_buffer_ = nullptr;
	OptixTraversableHandle ias_handle_{};

	LaunchParams h_launch_params_{};
	LaunchParams* d_launch_params_ = nullptr;
};