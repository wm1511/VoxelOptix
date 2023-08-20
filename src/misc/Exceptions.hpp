#pragma once

#define CCE(call) CheckResult<cudaError_t>("CUDA", (call), #call, __FILE__, __LINE__)
#define COE(call) CheckResult<OptixResult>("OPTIX", (call), #call, __FILE__, __LINE__)
#ifdef _DEBUG
#define CGE(call) call; CheckResult<GLenum>("OpenGL", glGetError(), #call, __FILE__, __LINE__)
#else
#define CGE(call) call
#endif

template <typename T>
void CheckResult(const char* library, const T result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::stringstream ss;
		ss << library << " error = " << static_cast<int>(result) << " at " << file << ": " << line << " '" << func << "'\n";
		throw std::exception(ss.str().c_str());
	}
}