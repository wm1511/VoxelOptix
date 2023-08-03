#include "App.hpp"

App::App() :
	window_(std::make_unique<Window>(1920, 1080, "Voxel Optix")),
	frame_(std::make_unique<Frame>(window_->GetWidth(), window_->GetHeight())),
	menu_(std::make_unique<Menu>()),
	camera_(std::make_shared<Camera>(window_->GetWidth(), window_->GetHeight(), 1.3f, 20.0f, 0.02f)),
	world_(std::make_shared<World>()),
	renderer_(std::make_unique<Renderer>(window_->GetWidth(), window_->GetHeight(), camera_, world_))
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

	menu_->CheckCursorMode(window_->GetGLFWWindow());

	camera_->Update(window_->GetGLFWWindow(), delta_time_, menu_->InMenu());

	float4* device_memory = frame_->MapMemory();
	renderer_->Render(device_memory, static_cast<float>(current_frame));
	frame_->UnmapMemory();

	frame_->Display();
}

void App::OnResize() const
{
	const int width = window_->GetWidth(), height = window_->GetHeight();

	camera_->HandleWindowResize(width, height);
	renderer_->HandleWindowResize(width, height);
	frame_->Recreate(width, height);
	window_->ResetResizedFlag();
}