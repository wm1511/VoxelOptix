#include "Menu.hpp"

Menu::Menu(std::shared_ptr<Renderer> renderer) :
	renderer_(std::move(renderer))
{
}

void Menu::CheckCursorMode(GLFWwindow* window)
{
	if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		{
			in_menu_ = true;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL)
	{
		if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS)
		{
			in_menu_ = false;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}
	}
}

void Menu::SwitchDenoiserState(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS)
	{
		if (renderer_->DenoiserActive())
			renderer_->DestroyDenoiser();
	}
	else if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS)
	{
		if (!renderer_->DenoiserActive())
			renderer_->InitDenoiser();
	}
}
