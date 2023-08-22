#include "Menu.hpp"

Menu::Menu(std::shared_ptr<Window> window, std::shared_ptr<Renderer> renderer, std::shared_ptr<World> world) :
	window_(std::move(window)),
	renderer_(std::move(renderer)),
	world_(std::move(world)),
	text_renderer_(std::make_unique<TextRenderer>())
{
}

void Menu::Update()
{
	ChangeSelection();

	if (selected_ == 0)
		HandleMenuExit();
	else if (selected_ == 1)
		SwitchDenoiserState();
	else if (selected_ == 2)
		SetValue(renderer_->GetMaxDepth(), 1u, 31u);
	else if (selected_ == 3)
	{
		if (SetValue(world_->GetGenerationDistance(), renderer_->GetRenderDistance() + 2, 12))
			world_->SetUpdateFlag();
	}
	else if (selected_ == 4)
	{
		if (SetValue(renderer_->GetRenderDistance(), 1, world_->GetGenerationDistance() - 2))
			world_->SetUpdateFlag();
	}
	else if (selected_ == 5)
		CloseApp();
}

void Menu::Display() const
{
	constexpr float3 black = { 0.0f, 0.0f, 0.0f };
	constexpr float3 white = { 1.0f, 1.0f, 1.0f };

	// Main text
	text_renderer_->Render(0.0f, 0.8f, "Voxel Optix", 24.0f, make_float3(0.0f));

	text_renderer_->Render(0.0f, 0.4f, "Start", 12.0f, selected_ == 0 ? white : black);

	std::string text = std::string("Denoiser: ") + (renderer_->DenoiserActive() ? "ON" : "OFF");
	text_renderer_->Render(0.0f, 0.2f, text, 12.0f, selected_ == 1 ? white : black);

	text = "Max trace depth: " + std::to_string(renderer_->GetMaxDepth());
	text_renderer_->Render(0.0f, 0.0f, text, 12.0f, selected_ == 2 ? white : black);

	text = "Generation distance: " + std::to_string(world_->GetGenerationDistance());
	text_renderer_->Render(0.0f, -0.2f, text, 12.0f, selected_ == 3 ? white : black);

	text = "Render distance: " + std::to_string(renderer_->GetRenderDistance());
	text_renderer_->Render(0.0f, -0.4f, text, 12.0f, selected_ == 4 ? white : black);

	text_renderer_->Render(0.0f, -0.6f, "Exit", 12.0f, selected_ == 5 ? white : black);

	// Comments
	if (world_->NeedsUpdate())
	{
		text = "World is being rebuilt, please wait...";
		text_renderer_->Render(0.0f, 0.3f, text, 4.0f, make_float3(1.0f, 0.0f, 0.0f));
	}

	if (selected_ == 1)
	{
		text = "Enables/disables Optix AI Denoiser. Affects tracing performance";
		text_renderer_->Render(0.0f, 0.1f, text, 4.0f, white);
	}
	else if (selected_ == 2)
	{
		text = "Sets maximal amount of ray reflections before its termination";
		text_renderer_->Render(0.0f, -0.1f, text, 4.0f, white);
	}
	else if (selected_ == 3)
	{
		text = "Sets world generation distance. Increasing takes a lot of time and GPU memory. Be careful, check GPU memory in Task Manager";
		text_renderer_->Render(0.0f, -0.3f, text, 4.0f, white);
	}
	else if (selected_ == 4)
	{
		text = "Sets world render distance. Affects tracing performance a bit. Increasing needs generation distance to be increased as well";
		text_renderer_->Render(0.0f, -0.5f, text, 4.0f, white);
	}
}

void Menu::HandleMenuEnter()
{
	GLFWwindow* window = window_->GetGLFWWindow();

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		in_menu_ = true;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

void Menu::HandleMenuExit()
{
	GLFWwindow* window = window_->GetGLFWWindow();

	if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS && !world_->NeedsUpdate())
	{
		in_menu_ = false;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
}

void Menu::SwitchDenoiserState()
{
	GLFWwindow* window = window_->GetGLFWWindow();

	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS && renderer_->DenoiserActive())
		renderer_->DestroyDenoiser();
	else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS && !renderer_->DenoiserActive())
		renderer_->InitDenoiser();
}

void Menu::ChangeSelection()
{
	GLFWwindow* window = window_->GetGLFWWindow();

	if (selected_ < 0)
		selected_ = 5;
	else if (selected_ > 5)
		selected_ = 0;

	const int down_state = glfwGetKey(window, GLFW_KEY_DOWN);
	if (down_state == GLFW_PRESS && last_down_state_ != GLFW_PRESS)
		selected_++;

	last_down_state_ = down_state;

	const int up_state = glfwGetKey(window, GLFW_KEY_UP);
	if (up_state == GLFW_PRESS && last_up_state_ != GLFW_PRESS)
		selected_--;

	last_up_state_ = up_state;
}

void Menu::CloseApp() const
{
	GLFWwindow* window = window_->GetGLFWWindow();

	if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS)
		window_->SetCloseFlag();
}
