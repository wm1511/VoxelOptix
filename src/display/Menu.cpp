#include "Menu.hpp"

Menu::Menu(std::shared_ptr<Window> window, std::shared_ptr<Renderer> renderer) :
	window_(std::move(window)),
	renderer_(std::move(renderer)),
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
		SetMaxDepth();
	else if (selected_ == 3)
		CloseApp();
}

void Menu::Display() const
{
	constexpr float3 black = { 0.0f, 0.0f, 0.0f };
	constexpr float3 white = { 1.0f, 1.0f, 1.0f };

	text_renderer_->Render(0.0f, 0.8f, "Voxel Optix", 24.0f, make_float3(0.0f));

	text_renderer_->Render(0.0f, 0.4f, "Start", 12.0f, selected_ == 0 ? white : black);

	std::string text = std::string("Denoiser: ") + (renderer_->DenoiserActive() ? "ON" : "OFF");
	text_renderer_->Render(0.0f, 0.2f, text, 12.0f, selected_ == 1 ? white : black);

	text = "Max trace depth: " + std::to_string(renderer_->GetMaxDepth());
	text_renderer_->Render(0.0f, 0.0f, text, 12.0f, selected_ == 2 ? white : black);

	text_renderer_->Render(0.0f, -0.2f, "Exit", 12.0f, selected_ == 3 ? white : black);
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

	if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS)
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
		selected_ = 3;
	else if (selected_ > 3)
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

void Menu::SetMaxDepth()
{
	GLFWwindow* window = window_->GetGLFWWindow();

	const unsigned max_depth = renderer_->GetMaxDepth();

	const int left_state = glfwGetKey(window, GLFW_KEY_LEFT);
	if (left_state == GLFW_PRESS && last_left_state_ != GLFW_PRESS && max_depth > 1)
		renderer_->SetMaxDepth(max_depth - 1);

	last_left_state_ = left_state;

	const int right_state = glfwGetKey(window, GLFW_KEY_RIGHT);
	if (right_state == GLFW_PRESS && last_right_state_ != GLFW_PRESS && max_depth < 31)
		renderer_->SetMaxDepth(max_depth + 1);

	last_right_state_ = right_state;
}

void Menu::CloseApp() const
{
	GLFWwindow* window = window_->GetGLFWWindow();

	if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS)
		window_->SetCloseFlag();
}
