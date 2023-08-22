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
	void HandleAsUpdate();
	void InitDenoiser();
	void DestroyDenoiser();

	[[nodiscard]] unsigned& GetMaxDepth() { return h_launch_params_.max_depth; }
	[[nodiscard]] int& GetRenderDistance() { return render_distance_; }
	[[nodiscard]] bool DenoiserActive() const { return denoiser_active_; }

private:
	void InitOptix();
	void CreateModules();
	void CreatePrograms();
	void CreatePipeline();
	void PrepareAs(const OptixBuildInput& build_input, void*& buffer, OptixTraversableHandle& handle, OptixBuildOperation operation, cudaStream_t stream) const;
	void PrepareGas(OptixBuildOperation operation);
	void PrepareTopIas(OptixBuildOperation operation);
	void PrepareBottomIas(void*& buffer, OptixTraversableHandle& handle, const Chunk& chunk) const;
	void CreateSbt();

	// "Foreign" pointers
	std::shared_ptr<Camera> camera_ = nullptr;
	std::shared_ptr<World> world_ = nullptr;

	// Core structures
	cudaStream_t main_stream_{}, bottom_ias_stream_{};
	OptixDeviceContext context_ = nullptr;
	OptixModule module_ = nullptr;
	OptixPipeline pipeline_ = nullptr;
	OptixShaderBindingTable sbt_{};
	OptixDenoiser denoiser_ = nullptr;

	// Options
	int render_distance_{4}, last_render_distance_{4};
	OptixPipelineCompileOptions pipeline_compile_options_{};

	// Denoiser
	void* denoiser_scratch_buffer_ = nullptr, * denoiser_state_buffer_ = nullptr;
	OptixDenoiserSizes denoiser_sizes_{};
	bool denoiser_active_{};

	// Programs
	std::vector<OptixProgramGroup> raygen_programs_{};
	std::vector<OptixProgramGroup> miss_programs_{};
	std::vector<OptixProgramGroup> hit_programs_{};

	// Sbt data
	SbtRecord<RayGenData>* d_raygen_records_ = nullptr;
	SbtRecord<MissData>* d_miss_records_ = nullptr;
	SbtRecord<HitGroupData>* d_hit_records_ = nullptr;

	// Acceleration structures
	void* top_ias_buffer_ = nullptr, * gas_buffer_ = nullptr;
	OptixTraversableHandle top_ias_handle_{}, gas_handle_{};
	std::vector<void*> bottom_ias_buffers_{};
	std::vector<OptixTraversableHandle> bottom_ias_handles_{};

	// Params
	LaunchParams h_launch_params_{};
	LaunchParams* d_launch_params_ = nullptr;
};