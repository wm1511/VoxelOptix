#pragma once
#include <exception>
#include <format>

#define CCE(val) check_result<cudaError_t>("CUDA", (val), #val, __FILE__, __LINE__)
#define COE(val) check_result<OptixResult>("OPTIX", (val), #val, __FILE__, __LINE__)

template <typename T>
void check_result(const char* library, const T result, char const* const func, const char* const file, int const line)
{
	if (result)
		throw std::exception(std::format("{} error = {} at {}: {} '{}'", library, static_cast<int>(result), file, line, func).c_str());
}