#pragma once
#include "TextRenderer.hpp"
#include "Window.hpp"
#include "../render/Renderer.hpp"

class Menu
{
public:
	Menu(std::shared_ptr<Window> window, std::shared_ptr<Renderer> renderer);

	void Update();
	void Display() const;
	void HandleMenuEnter();
	void HandleMenuExit();

	[[nodiscard]] bool InMenu() const { return in_menu_; }

private:
	void SwitchDenoiserState();
	void ChangeSelection();
	void SetMaxDepth();
	void CloseApp() const;

	std::shared_ptr<Window> window_ = nullptr;
	std::shared_ptr<Renderer> renderer_ = nullptr;
	std::unique_ptr<TextRenderer> text_renderer_ = nullptr;

	int selected_{};
	int last_left_state_{}, last_up_state_{}, last_right_state_{}, last_down_state_{};
	bool in_menu_ = true;
};