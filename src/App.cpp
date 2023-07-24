#include "App.hpp"
#include "Utils.hpp"

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

	frame_->MapMemory();

	// Render here
	CCE(cudaMemset(frame_->GetMemory(), 1, frame_->GetSize()));

	frame_->UnmapMemory();

	frame_->Display();
}

void App::OnResize() const
{
	frame_->Recreate(window_->GetWidth(), window_->GetHeight());
	window_->ResetResizedFlag();
}