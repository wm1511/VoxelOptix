#include "Renderer.hpp"
#include "Exceptions.hpp"

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#ifdef _DEBUG
static void context_log(const unsigned int level, const char* tag, const char* message, void*)
{
	printf("[%u][%s]: %s\n", level, tag, message);
}
#endif

static std::string read_shader(const std::string& program_name)
{
	const std::filesystem::path path = std::filesystem::current_path() / "CMakeFiles" / "OptixPTX.dir" / "src" / program_name;

	std::ifstream file(path, std::ios::in | std::ios::binary);

	if (!file)
		throw std::exception("Failed to open Optix PTX shader file");

    const size_t size = file_size(path);
    std::string source(size, '\0');
	
    file.read(source.data(), static_cast<long long>(size));
	file.close();

	return source;
}

static void fill_matrix(float matrix[12], const float3 t, const float3 s, const float3 r)
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

Renderer::Renderer()
{
	init_optix();
	create_modules();
	create_programs();
	create_pipeline();

	gas_handles_.resize(1);
	gas_buffers_.resize(1);

	prepare_gas(gas_handles_[0], gas_buffers_[0], OPTIX_BUILD_OPERATION_BUILD);
	prepare_ias(gas_handles_, OPTIX_BUILD_OPERATION_BUILD);

	create_sbt();

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_launch_params_), sizeof(LaunchParams)));
	CCE(cudaMemcpy(d_launch_params_, &h_launch_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

Renderer::~Renderer()
{
	CCE(cudaFree(d_launch_params_));

	CCE(cudaFree(d_raygen_records_));
	CCE(cudaFree(d_miss_records_));
	CCE(cudaFree(d_hit_records_));

	CCE(cudaFree(ias_buffer_));
	CCE(cudaFree(gas_buffers_[0]));

	gas_buffers_.clear();
	gas_handles_.clear();

	CCE(cudaStreamDestroy(stream_));
	COE(optixDeviceContextDestroy(context_));
}

void Renderer::render(float4* device_memory, const int width, const int height)
{
	if (width < 64 || height < 64)
		return;

	h_launch_params_.width = width;
	h_launch_params_.height = height;
	h_launch_params_.frame_buffer = device_memory;
	h_launch_params_.traversable = ias_handle_;
	h_launch_params_.camera_info.update(static_cast<float>(width), static_cast<float>(height));

	CCE(cudaMemcpy(d_launch_params_, &h_launch_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));

	COE(optixLaunch(
	    pipeline_, stream_,
	    reinterpret_cast<CUdeviceptr>(d_launch_params_),
	    sizeof(LaunchParams),
	    &sbt_,
	    h_launch_params_.width,
	    h_launch_params_.height,
	    1));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());
}

void Renderer::init_optix()
{
	COE(optixInit());

	OptixDeviceContextOptions options{};

#ifdef _DEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
	options.logCallbackFunction = &context_log;
	options.logCallbackLevel = 4;
#endif

	const CUcontext cuda_context = nullptr;

	CCE(cudaStreamCreate(&stream_));
	COE(optixDeviceContextCreate(cuda_context, &options, &context_));
}

void Renderer::create_modules()
{
	module_compile_options_.maxRegisterCount = 50;
#ifdef _DEBUG
	module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
	module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

	pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
	pipeline_compile_options_.usesMotionBlur = false;
	pipeline_compile_options_.numPayloadValues = 3;
	pipeline_compile_options_.numAttributeValues = 0;
#ifdef _DEBUG
	pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
#else
	pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
	pipeline_compile_options_.pipelineLaunchParamsVariableName = "launch_params";

	const std::string shader = read_shader("Programs.ptx");

	COE(optixModuleCreate(
		context_,
		&module_compile_options_,
		&pipeline_compile_options_,
		shader.c_str(),
		shader.size(),
		nullptr, nullptr,
		&module_));
}

void Renderer::create_programs()
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

void Renderer::create_pipeline()
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

	constexpr OptixPipelineLinkOptions pipeline_link_options{ 8u };

	COE(optixPipelineCreate(
		context_,
		&pipeline_compile_options_,
		&pipeline_link_options,
		program_groups.data(),
		static_cast<unsigned>(program_groups.size()),
		nullptr, nullptr,
		&pipeline_));

	COE(optixPipelineSetStackSize(pipeline_, 2 * 1024, 2 * 1024, 2 * 1024, 2));
}

void Renderer::prepare_as(const OptixBuildInput& build_input, void*& buffer, OptixTraversableHandle& handle, const OptixBuildOperation operation) const
{
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
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
			nullptr,
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
			nullptr,
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
			nullptr,
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

void Renderer::prepare_gas(OptixTraversableHandle& handle, void*& buffer, const OptixBuildOperation operation) const
{
	unsigned indices[]
	{
        2, 6, 7,
        2, 3, 7,

        0, 4, 5,
        0, 1, 5,

        0, 2, 6,
        0, 4, 6,

        1, 3, 7,
        1, 5, 7,

        0, 2, 3,
        0, 1, 3,

        4, 6, 7,
        4, 5, 7
    };


    float vertices[]
	{
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f 
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
	input.triangleArray.numVertices = static_cast<unsigned>(8);
	input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&device_vertices);

	input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	input.triangleArray.indexStrideInBytes = sizeof(uint3);
	input.triangleArray.numIndexTriplets = static_cast<unsigned>(12);
	input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(device_indices);

	input.triangleArray.flags = flags;
	input.triangleArray.numSbtRecords = 1;
	input.triangleArray.sbtIndexOffsetBuffer = 0;
	input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	prepare_as(input, buffer, handle, operation);
}

void Renderer::prepare_ias(std::vector<OptixTraversableHandle>& gases, const OptixBuildOperation operation)
{
    std::vector<OptixInstance> instances;

	float transform[12] = {	1.0f, 0.0f, 0.0f, 0.0f, 
							0.0f, 1.0f, 0.0f, 0.0f, 
							0.0f, 0.0f, 1.0f, 0.0f };

	for (unsigned i = 0; i < gases.size(); i++)
    {
        OptixInstance instance = {};
        
        instance.instanceId = 0;
        instance.visibilityMask = 255;
        instance.sbtOffset = i;
        instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;

		memcpy_s(instance.transform, sizeof instance.transform, transform, sizeof transform);

        instance.traversableHandle = gases[i];
        instances.push_back(instance);
    }

    OptixInstance* instance_buffer;
	CCE(cudaMalloc(reinterpret_cast<void**>(&instance_buffer), instances.size() * sizeof(OptixInstance)));
	CCE(cudaMemcpy(instance_buffer, instances.data(), instances.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));

    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(instance_buffer);
    input.instanceArray.numInstances = static_cast<unsigned>(instances.size());

	prepare_as(input, ias_buffer_, ias_handle_, operation);
    
	CCE(cudaFree(instance_buffer));
}

void Renderer::create_sbt()
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