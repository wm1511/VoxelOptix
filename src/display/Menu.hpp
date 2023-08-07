#pragma once
#include "TextRenderer.hpp"
#include "../render/Renderer.hpp"

class Menu
{
public:
	explicit Menu(std::shared_ptr<Renderer> renderer);

	void Display() const;

	void CheckCursorMode(GLFWwindow* window);
	void SwitchDenoiserState(GLFWwindow* window);
	[[nodiscard]] bool InMenu() const { return in_menu_; }

private:
	std::shared_ptr<Renderer> renderer_ = nullptr;
	std::unique_ptr<TextRenderer> text_renderer_ = nullptr;

	bool in_menu_ = true;
};