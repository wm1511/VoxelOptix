#include "Renderer.hpp"
#include "../misc/Exceptions.hpp"

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

namespace
{
#ifdef _DEBUG
	void ContextLog(const unsigned int level, const char* tag, const char* message, void*)
	{
		printf("[%u][%s]: %s\n", level, tag, message);
	}
#endif

	std::string ReadShader(const std::string& program_name)
	{
		const std::filesystem::path path = std::filesystem::current_path() / "CMakeFiles" / "OptixPTX.dir" / "src" / "render" / program_name;

		std::ifstream file(path, std::ios::in | std::ios::binary);

		if (!file)
			throw std::exception("Failed to open Optix PTX shader file");

		const size_t size = file_size(path);
		std::string source(size, '\0');

		file.read(source.data(), static_cast<long long>(size));
		file.close();

		return source;
	}

	void FillMatrix(float matrix[12], const float3 t = { 0.0f, 0.0f, 0.0f }, const float3 s = { 1.0f, 1.0f, 1.0f }, const float3 r = { 0.0f, 0.0f, 0.0f })
	{
		const float sa = sin(r.z);
		const float ca = cos(r.z);
		const float sb = sin(r.y);
		const float cb = cos(r.y);
		const float sc = sin(r.x);
		const float cc = cos(r.x);

		matrix[0] = cb * cc * s.x;
		matrix[1] = -cb * sc * s.y;
		matrix[2] = sb * s.z;
		matrix[3] = t.x;
		matrix[4] = (sa * sb * cc + ca * sc) * s.x;
		matrix[5] = (-sa * sb * sc + ca * cc) * s.y;
		matrix[6] = -sa * cb * s.z;
		matrix[7] = t.y;
		matrix[8] = (-ca * sb * cc + sa * sc) * s.x;
		matrix[9] = (ca * sb * sc + sa * cc) * s.y;
		matrix[10] = ca * cb * s.z;
		matrix[11] = t.z;
	}
}

Renderer::Renderer(const int width, const int height, std::shared_ptr<CameraController> camera_controller, std::shared_ptr<World> world) :
	camera_controller_(std::move(camera_controller)),
	world_(std::move(world))
{
	InitOptix();
	CreateModules();
	CreatePrograms();
	CreatePipeline();

	PrepareGas(OPTIX_BUILD_OPERATION_BUILD);

	for (auto& chunk : world_->GetChunks() | std::views::values)
		PrepareBottomIas(chunk.GetIasBuffer(), chunk.GetIasHandle(), chunk);

	PrepareTopIas(OPTIX_BUILD_OPERATION_BUILD);

	CreateSbt();

	h_launch_params_.width = width;
	h_launch_params_.height = height;

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_launch_params_), sizeof(LaunchParams)));
}

Renderer::~Renderer()
{
	CCE(cudaFree(d_launch_params_));

	CCE(cudaFree(d_raygen_records_));
	CCE(cudaFree(d_miss_records_));
	CCE(cudaFree(d_hit_records_));

	CCE(cudaFree(top_ias_buffer_));
	CCE(cudaFree(gas_buffer_));

	CCE(cudaStreamDestroy(bottom_ias_stream_));
	CCE(cudaStreamDestroy(main_stream_));
	COE(optixDeviceContextDestroy(context_));
}

void Renderer::Render(float4* frame_pointer, const float time)
{
	if (h_launch_params_.width < 64 || h_launch_params_.height < 64)
		return;

	h_launch_params_.time = time;
	h_launch_params_.frame_pointer = frame_pointer;
	h_launch_params_.traversable = top_ias_handle_;

	camera_controller_->CalculateMapping(static_cast<float>(h_launch_params_.width) / static_cast<float>(h_launch_params_.height),
		h_launch_params_.camera.starting_point, h_launch_params_.camera.horizontal_map,
		h_launch_params_.camera.vertical_map, h_launch_params_.camera.position);

	CCE(cudaMemcpy(d_launch_params_, &h_launch_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));

	COE(optixLaunch(
		pipeline_, main_stream_,
		reinterpret_cast<CUdeviceptr>(d_launch_params_),
		sizeof(LaunchParams),
		&sbt_,
		h_launch_params_.width,
		h_launch_params_.height,
		1));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());
}

void Renderer::Denoise(float4* device_memory)
{
	OptixDenoiserParams denoiser_params{};
	OptixDenoiserLayer layer{};

	layer.input.data = reinterpret_cast<CUdeviceptr>(device_memory);
	layer.input.width = h_launch_params_.width;
	layer.input.height = h_launch_params_.height;
	layer.input.rowStrideInBytes = h_launch_params_.width * sizeof(float4);
	layer.input.pixelStrideInBytes = sizeof(float4);
	layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

	layer.output.data = reinterpret_cast<CUdeviceptr>(device_memory);
	layer.output.width = h_launch_params_.width;
	layer.output.height = h_launch_params_.height;
	layer.output.rowStrideInBytes = h_launch_params_.width * sizeof(float4);
	layer.output.pixelStrideInBytes = sizeof(float4);
	layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

	constexpr OptixDenoiserGuideLayer guide_layer{};

	COE(optixDenoiserInvoke(denoiser_,
		main_stream_,
		&denoiser_params,
		reinterpret_cast<CUdeviceptr>(denoiser_state_buffer_),
		denoiser_sizes_.stateSizeInBytes,
		&guide_layer,
		&layer,
		1,
		0,
		0,
		reinterpret_cast<CUdeviceptr>(denoiser_scratch_buffer_),
		denoiser_sizes_.withoutOverlapScratchSizeInBytes));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());
}

void Renderer::HandleWindowResize(const int width, const int height)
{
	h_launch_params_.width = width;
	h_launch_params_.height = height;

	if (denoiser_active_)
	{
		DestroyDenoiser();
		InitDenoiser();
	}
}

void Renderer::HandleAsUpdate()
{
	for (auto& chunk : world_->GetChunks() | std::views::values)
		if (chunk.GetIasHandle() == 0)
			PrepareBottomIas(chunk.GetIasBuffer(), chunk.GetIasHandle(), chunk);

	CCE(cudaStreamSynchronize(bottom_ias_stream_));

	if (render_distance_ == last_render_distance_)
	{
		PrepareTopIas(OPTIX_BUILD_OPERATION_UPDATE);
	}
	else
	{
		PrepareTopIas(OPTIX_BUILD_OPERATION_BUILD);
		last_render_distance_ = render_distance_;
	}

	world_->ResetUpdateFlag();
}

void Renderer::InitOptix()
{
	COE(optixInit());

	OptixDeviceContextOptions options{};

#ifdef _DEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
	options.logCallbackFunction = &ContextLog;
	options.logCallbackLevel = 4;
#endif

	const CUcontext cuda_context = nullptr;

	CCE(cudaStreamCreate(&main_stream_));
	CCE(cudaStreamCreate(&bottom_ias_stream_));
	COE(optixDeviceContextCreate(cuda_context, &options, &context_));
}

void Renderer::InitDenoiser()
{
	OptixDenoiserOptions denoiser_options = {};

	COE(optixDenoiserCreate(context_, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiser_options, &denoiser_));

	COE(optixDenoiserComputeMemoryResources(denoiser_, h_launch_params_.width, h_launch_params_.height, &denoiser_sizes_));

	CCE(cudaMalloc(&denoiser_state_buffer_, denoiser_sizes_.stateSizeInBytes));
	CCE(cudaMalloc(&denoiser_scratch_buffer_, denoiser_sizes_.withoutOverlapScratchSizeInBytes));

	COE(optixDenoiserSetup(
		denoiser_,
		main_stream_,
		h_launch_params_.width,
		h_launch_params_.height,
		reinterpret_cast<CUdeviceptr>(denoiser_state_buffer_),
		denoiser_sizes_.stateSizeInBytes,
		reinterpret_cast<CUdeviceptr>(denoiser_scratch_buffer_),
		denoiser_sizes_.withoutOverlapScratchSizeInBytes));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());

	denoiser_active_ = true;
}

void Renderer::DestroyDenoiser()
{
	denoiser_active_ = false;

	CCE(cudaFree(denoiser_state_buffer_));
	CCE(cudaFree(denoiser_scratch_buffer_));
	COE(optixDenoiserDestroy(denoiser_));
}

void Renderer::CreateModules()
{
	OptixModuleCompileOptions module_compile_options{};

	module_compile_options.maxRegisterCount = 50;
#ifdef _DEBUG
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

	pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
	pipeline_compile_options_.usesMotionBlur = false;
	pipeline_compile_options_.numPayloadValues = 11;
	pipeline_compile_options_.numAttributeValues = 0;
#ifdef _DEBUG
	pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
#else
	pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
	pipeline_compile_options_.pipelineLaunchParamsVariableName = "launch_params";

	const std::string shader = ReadShader("Programs.ptx");

	COE(optixModuleCreate(
		context_,
		&module_compile_options,
		&pipeline_compile_options_,
		shader.c_str(),
		shader.size(),
		nullptr, nullptr,
		&module_));
}

void Renderer::CreatePrograms()
{
	raygen_programs_.resize(1);
	OptixProgramGroupOptions rg_options = {};
	OptixProgramGroupDesc rg_desc = {};
	rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	rg_desc.raygen.module = module_;
	rg_desc.raygen.entryFunctionName = "__raygen__render";

	COE(optixProgramGroupCreate(
		context_,
		&rg_desc,
		1,
		&rg_options,
		nullptr, nullptr,
		raygen_programs_.data()));

	miss_programs_.resize(1);
	OptixProgramGroupOptions m_options = {};
	OptixProgramGroupDesc m_desc = {};
	m_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	m_desc.miss.module = module_;
	m_desc.miss.entryFunctionName = "__miss__sky";

	COE(optixProgramGroupCreate(
		context_,
		&m_desc,
		1,
		&m_options,
		nullptr, nullptr,
		miss_programs_.data()));

	hit_programs_.resize(1);
	OptixProgramGroupOptions hg_options = {};
	OptixProgramGroupDesc hg_desc = {};
	hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hg_desc.hitgroup.moduleCH = module_;
	hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";

	COE(optixProgramGroupCreate(
		context_,
		&hg_desc,
		1,
		&hg_options,
		nullptr, nullptr,
		hit_programs_.data()));
}

void Renderer::CreatePipeline()
{
	std::vector<OptixProgramGroup> program_groups;

	program_groups.reserve(program_groups.size() + raygen_programs_.size());
	for (auto pg : raygen_programs_)
		program_groups.push_back(pg);

	program_groups.reserve(program_groups.size() + miss_programs_.size());
	for (auto pg : miss_programs_)
		program_groups.push_back(pg);

	program_groups.reserve(program_groups.size() + hit_programs_.size());
	for (auto pg : hit_programs_)
		program_groups.push_back(pg);

	const OptixPipelineLinkOptions pipeline_link_options{ h_launch_params_.max_depth };

	COE(optixPipelineCreate(
		context_,
		&pipeline_compile_options_,
		&pipeline_link_options,
		program_groups.data(),
		static_cast<unsigned>(program_groups.size()),
		nullptr, nullptr,
		&pipeline_));

	COE(optixPipelineSetStackSize(pipeline_, 2 * 1024, 2 * 1024, 2 * 1024, 3));
}

void Renderer::PrepareAs(const OptixBuildInput& build_input, void*& buffer, OptixTraversableHandle& handle, const OptixBuildOperation operation, const cudaStream_t stream) const
{
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.motionOptions.numKeys = 1;
	accel_options.operation = operation;

	OptixAccelBufferSizes buffer_sizes;
	COE(optixAccelComputeMemoryUsage(
		context_,
		&accel_options,
		&build_input,
		1,
		&buffer_sizes));

	void* temp_buffer = nullptr;

	if (operation == OPTIX_BUILD_OPERATION_BUILD)
	{
		uint64_t* compacted_size_buffer;
		CCE(cudaMalloc(reinterpret_cast<void**>(&compacted_size_buffer), sizeof(uint64_t)));

		OptixAccelEmitDesc emit_desc;
		emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emit_desc.result = reinterpret_cast<CUdeviceptr>(compacted_size_buffer);

		CCE(cudaMalloc(&temp_buffer, buffer_sizes.tempSizeInBytes));

		void* output_buffer;
		CCE(cudaMalloc(&output_buffer, buffer_sizes.outputSizeInBytes));

		COE(optixAccelBuild(
			context_,
			stream,
			&accel_options,
			&build_input,
			1,
			reinterpret_cast<CUdeviceptr>(temp_buffer),
			buffer_sizes.tempSizeInBytes,
			reinterpret_cast<CUdeviceptr>(output_buffer),
			buffer_sizes.outputSizeInBytes,
			&handle,
			&emit_desc, 1));

		CCE(cudaDeviceSynchronize());
		CCE(cudaGetLastError());

		uint64_t compacted_size;
		CCE(cudaMemcpy(&compacted_size, compacted_size_buffer, sizeof(uint64_t), cudaMemcpyDeviceToHost));

		CCE(cudaMalloc(&buffer, compacted_size));
		COE(optixAccelCompact(
			context_,
			stream,
			handle,
			reinterpret_cast<CUdeviceptr>(buffer),
			compacted_size,
			&handle));

		CCE(cudaDeviceSynchronize());
		CCE(cudaGetLastError());

		CCE(cudaFree(output_buffer));
		CCE(cudaFree(compacted_size_buffer));
	}
	else
	{
		CCE(cudaMalloc(&temp_buffer, buffer_sizes.tempUpdateSizeInBytes));

		COE(optixAccelBuild(
			context_,
			stream,
			&accel_options,
			&build_input,
			1,
			reinterpret_cast<CUdeviceptr>(temp_buffer),
			buffer_sizes.tempUpdateSizeInBytes,
			reinterpret_cast<CUdeviceptr>(buffer),
			buffer_sizes.outputSizeInBytes,
			&handle,
			nullptr, 0));

		CCE(cudaDeviceSynchronize());
		CCE(cudaGetLastError());
	}

	CCE(cudaFree(temp_buffer));
}

void Renderer::PrepareGas(const OptixBuildOperation operation)
{
	unsigned indices[]
	{
		// Top
		2, 6, 7,
		2, 3, 7,

		// Bottom
		0, 4, 5,
		0, 1, 5,

		// Left
		0, 2, 6,
		0, 4, 6,

		// Right
		1, 3, 7,
		1, 5, 7,

		// Front
		0, 2, 3,
		0, 1, 3,

		// Back
		4, 6, 7,
		4, 5, 7
	};


	float vertices[]
	{
		-0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		-0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		-0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f
	};

	unsigned* device_indices = nullptr;
	float* device_vertices = nullptr;

	CCE(cudaMalloc(reinterpret_cast<void**>(&device_indices), sizeof indices));
	CCE(cudaMalloc(reinterpret_cast<void**>(&device_vertices), sizeof vertices));
	CCE(cudaMemcpy(device_vertices, vertices, sizeof vertices, cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(device_indices, indices, sizeof indices, cudaMemcpyHostToDevice));

	OptixBuildInput input{};
	unsigned flags[1] = { OPTIX_BUILD_FLAG_NONE };

	input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	input.triangleArray.vertexStrideInBytes = sizeof(float3);
	input.triangleArray.numVertices = 8u;
	input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&device_vertices);

	input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	input.triangleArray.indexStrideInBytes = sizeof(uint3);
	input.triangleArray.numIndexTriplets = 12u;
	input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(device_indices);

	input.triangleArray.flags = flags;
	input.triangleArray.numSbtRecords = 1;
	input.triangleArray.sbtIndexOffsetBuffer = 0;
	input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	PrepareAs(input, gas_buffer_, gas_handle_, operation, main_stream_);
}

void Renderer::PrepareTopIas(const OptixBuildOperation operation)
{
	std::vector<OptixInstance> instances;

	auto [camera_x, camera_y, camera_z] = world_->GetCameraChunk();

	int min_height = camera_y > -2 ? -render_distance_ : camera_y - render_distance_ / 2;
	int max_height = camera_y > -2 ? 0 : camera_y + render_distance_ / 2;

	for (int x = camera_x - render_distance_; x < camera_x + render_distance_; x++)
	{
		for (int y = min_height; y < max_height; y++)
		{
			for (int z = camera_z - render_distance_; z < camera_z + render_distance_; z++)
			{
				OptixInstance instance = {};

				instance.instanceId = static_cast<unsigned>(instances.size());
				instance.visibilityMask = 255;
				instance.sbtOffset = 0;
				instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;

				FillMatrix(instance.transform, make_float3(0.0f));

				instance.traversableHandle = world_->GetChunk(make_int3(x, y, z)).GetIasHandle();
				instances.push_back(instance);
			}
		}
	}

	OptixInstance* instance_buffer;
	CCE(cudaMalloc(reinterpret_cast<void**>(&instance_buffer), instances.size() * sizeof(OptixInstance)));
	CCE(cudaMemcpy(instance_buffer, instances.data(), instances.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));

	OptixBuildInput input = {};
	input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(instance_buffer);
	input.instanceArray.numInstances = static_cast<unsigned>(instances.size());

	PrepareAs(input, top_ias_buffer_, top_ias_handle_, operation, main_stream_);

	CCE(cudaFree(instance_buffer));
}

void Renderer::PrepareBottomIas(void*& buffer, OptixTraversableHandle& handle, const Chunk& chunk) const
{
	std::vector<OptixInstance> instances;

	for (unsigned char i = 0; i < Chunk::size_; i++)
	{
		for (unsigned char j = 0; j < Chunk::size_; j++)
		{
			for (unsigned char k = 0; k < Chunk::size_; k++)
			{
				if (!chunk.GetVoxel(i, j, k))
					continue;

				OptixInstance instance = {};

				instance.instanceId = static_cast<unsigned>(instances.size());
				instance.visibilityMask = 255;
				instance.sbtOffset = 0;
				instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;

				FillMatrix(instance.transform, chunk.GetPosition() + make_float3(i, j, k));

				instance.traversableHandle = gas_handle_;
				instances.push_back(instance);
			}
		}
	}

	OptixInstance* instance_buffer;
	CCE(cudaMalloc(reinterpret_cast<void**>(&instance_buffer), instances.size() * sizeof(OptixInstance)));
	CCE(cudaMemcpy(instance_buffer, instances.data(), instances.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));

	OptixBuildInput input = {};
	input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(instance_buffer);
	input.instanceArray.numInstances = static_cast<unsigned>(instances.size());

	PrepareAs(input, buffer, handle, OPTIX_BUILD_OPERATION_BUILD, bottom_ias_stream_);

	CCE(cudaFree(instance_buffer));
}

void Renderer::CreateSbt()
{
	std::vector<SbtRecord<RayGenData>> raygen_records;
	for (const auto& raygen_program : raygen_programs_)
	{
		SbtRecord<RayGenData> rec{};
		COE(optixSbtRecordPackHeader(raygen_program, &rec));
		raygen_records.push_back(rec);
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_raygen_records_), raygen_records.size() * sizeof(SbtRecord<RayGenData>)));
	CCE(cudaMemcpy(d_raygen_records_, raygen_records.data(), raygen_records.size() * sizeof(SbtRecord<RayGenData>), cudaMemcpyHostToDevice));

	sbt_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_records_);

	std::vector<SbtRecord<MissData>> miss_records;
	for (const auto& miss_program : miss_programs_)
	{
		SbtRecord<MissData> rec{};
		COE(optixSbtRecordPackHeader(miss_program, &rec));
		miss_records.push_back(rec);
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_miss_records_), miss_records.size() * sizeof(SbtRecord<MissData>)));
	CCE(cudaMemcpy(d_miss_records_, miss_records.data(), miss_records.size() * sizeof(SbtRecord<MissData>), cudaMemcpyHostToDevice));

	sbt_.missRecordBase = reinterpret_cast<CUdeviceptr>(d_miss_records_);
	sbt_.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
	sbt_.missRecordCount = static_cast<unsigned>(miss_records.size());

	std::vector<SbtRecord<HitGroupData>> hitgroup_records;
	for (const auto& hit_program : hit_programs_)
	{
		SbtRecord<HitGroupData> rec{};
		COE(optixSbtRecordPackHeader(hit_program, &rec));
		hitgroup_records.push_back(rec);
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_hit_records_), hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>)));
	CCE(cudaMemcpy(d_hit_records_, hitgroup_records.data(), hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>), cudaMemcpyHostToDevice));

	sbt_.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_hit_records_);
	sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitGroupData>);
	sbt_.hitgroupRecordCount = static_cast<unsigned>(hitgroup_records.size());
}
