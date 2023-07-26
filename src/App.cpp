#include "App.hpp"
#include "Utils.hpp"

extern void launch_kernel(float4* device_memory, int width, int height);

App::App() :
	window_(std::make_unique<Window>(1920, 1080, "Voxel Optix")),
	frame_(std::make_unique<Frame>(window_->GetWidth(), window_->GetHeight()))
{
}

void App::Run()
{
	while (!window_->ShouldClose())
	{
		glfwPollEvents();

		if (window_->Resized())
			OnResize();

		OnUpdate();

		glfwSwapBuffers(window_->GetGLFWWindow());
	}
}

void App::OnUpdate()
{
	const double current_frame = glfwGetTime();
	delta_time_ = current_frame - last_frame_;
	last_frame_ = current_frame;

	float4* device_memory = frame_->MapMemory();
	launch_kernel(device_memory, window_->GetWidth(), window_->GetHeight());
	frame_->UnmapMemory();

	frame_->Display();
}

void App::OnResize() const
{
	frame_->Recreate(window_->GetWidth(), window_->GetHeight());
	window_->ResetResizedFlag();
}