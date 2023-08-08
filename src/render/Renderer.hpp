#pragma once
#include "Camera.hpp"
#include "../world/World.hpp"
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
	Renderer(int width, int height, std::shared_ptr<Camera> camera, std::shared_ptr<World> world);
	~Renderer();

	Renderer(const Renderer&) = delete;
	Renderer(Renderer&&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	Renderer& operator=(Renderer&&) = delete;

	void Render(float4* frame_pointer, float time);
	void Denoise(float4* device_memory);
	void HandleWindowResize(int width, int height);
	void HandleIasRebuild();
	void InitDenoiser();
	void DestroyDenoiser();

	void SetMaxDepth(const unsigned depth) { h_launch_params_.max_depth = depth; }
	[[nodiscard]] unsigned GetMaxDepth() const { return h_launch_params_.max_depth; }
	[[nodiscard]] bool DenoiserActive() const { return denoiser_active_; }

private:
	void InitOptix();
	void CreateModules();
	void CreatePrograms();
	void CreatePipeline();
	void PrepareAs(const OptixBuildInput& build_input, void*& buffer, OptixTraversableHandle& handle, OptixBuildOperation operation) const;
	void PrepareGas(OptixBuildOperation operation);
	void PrepareIas(OptixBuildOperation operation);
	void CreateSbt();

	std::shared_ptr<Camera> camera_ = nullptr;
	std::shared_ptr<World> world_ = nullptr;

	cudaStream_t stream_{};
	OptixDeviceContext context_ = nullptr;
	OptixModule module_ = nullptr;
	OptixPipeline pipeline_ = nullptr;
	OptixShaderBindingTable sbt_{};
	OptixDenoiser denoiser_ = nullptr;

	OptixPipelineCompileOptions pipeline_compile_options_{};
	OptixModuleCompileOptions module_compile_options_{};
	OptixDenoiserParams denoiser_params_{};

	void* denoiser_scratch_buffer_ = nullptr, * denoiser_state_buffer_ = nullptr;
	OptixDenoiserSizes denoiser_sizes_{};
	bool denoiser_active_{};

	std::vector<OptixProgramGroup> raygen_programs_{};
	std::vector<OptixProgramGroup> miss_programs_{};
	std::vector<OptixProgramGroup> hit_programs_{};

	SbtRecord<RayGenData>* d_raygen_records_ = nullptr;
	SbtRecord<MissData>* d_miss_records_ = nullptr;
	SbtRecord<HitGroupData>* d_hit_records_ = nullptr;

	void* ias_buffer_ = nullptr, * gas_buffer_ = nullptr;
	OptixTraversableHandle ias_handle_{}, gas_handle_{};

	LaunchParams h_launch_params_{};
	LaunchParams* d_launch_params_ = nullptr;
};