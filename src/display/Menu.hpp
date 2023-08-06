#pragma once
#include "../render/Renderer.hpp"

class Menu
{
public:
	explicit Menu(std::shared_ptr<Renderer> renderer);

	void CheckCursorMode(GLFWwindow* window);
	void SwitchDenoiserState(GLFWwindow* window);
	[[nodiscard]] bool InMenu() const { return in_menu_; }

private:
	std::shared_ptr<Renderer> renderer_ = nullptr;

	bool in_menu_ = true;
};