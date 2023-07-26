#pragma once

#define CCE(call) check_result<cudaError_t>("CUDA", (call), #call, __FILE__, __LINE__)
#define COE(call) check_result<OptixResult>("OPTIX", (call), #call, __FILE__, __LINE__)
#ifdef _DEBUG
#define CGE(call) call; check_result<GLenum>("OpenGL", glGetError(), #call, __FILE__, __LINE__)
#else
#define CGE(call) call
#endif

template <typename T>
void check_result(const char* library, const T result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::stringstream ss;
		ss << library << " error = " << static_cast<int>(result) << " at " << file << ": " << line << " '" << func << "'\n";
		throw std::exception(ss.str().c_str());
	}
}