#pragma once
#include <exception>
#include <format>

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
		throw std::exception(std::format("{} error = {} at {}: {} '{}'", library, static_cast<int>(result), file, line, func).c_str());
}