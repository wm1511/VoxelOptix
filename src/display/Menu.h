#pragma once

class Menu
{
public:
	void CheckCursorMode(GLFWwindow* window);
	[[nodiscard]] bool InMenu() const { return in_menu_; }

private:
	bool in_menu_ = true;

};