#include "Menu.h"

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
