#include "App.hpp"

App::App() :
	window_(std::make_unique<Window>(1920, 1080, "Voxel Optix")),
	frame_(std::make_unique<Frame>(window_->GetWidth(), window_->GetHeight())),
	camera_(std::make_shared<Camera>(window_->GetWidth(), window_->GetHeight())),
	world_(std::make_shared<World>()),
	renderer_(std::make_shared<Renderer>(window_->GetWidth(), window_->GetHeight(), camera_, world_)),
	menu_(std::make_unique<Menu>(renderer_))
{
}

void App::Run()
{
	while (!window_->ShouldClose())
	{
		glfwPollEvents();

		if (window_->Resized())
			OnResize();

		if (static_cast<unsigned>(glfwGetTime()) != static_cast<unsigned>(last_frame_))
			OnceASecond();

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
	menu_->SwitchDenoiserState(window_->GetGLFWWindow());

	camera_->Update(window_->GetGLFWWindow(), delta_time_, menu_->InMenu());

	float4* frame_pointer = frame_->MapMemory();

	renderer_->Render(frame_pointer, static_cast<float>(current_frame));

	if (renderer_->DenoiserActive())
		renderer_->Denoise(frame_pointer);

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

void App::OnceASecond() const
{
	window_->SetTitle(std::format("Voxel Optix - FPS: {}", static_cast<unsigned>(1.0 / delta_time_)));
}
