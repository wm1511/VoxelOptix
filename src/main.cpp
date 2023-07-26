#include "App.hpp"

int main()
{
	try
	{
		App app;
		app.Run();
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		cudaDeviceReset();
		return EXIT_FAILURE;
	}
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
