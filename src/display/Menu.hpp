#pragma once
#include "TextRenderer.hpp"
#include "Window.hpp"
#include "../render/Renderer.hpp"

class Menu
{
public:
	Menu(std::shared_ptr<Window> window, std::shared_ptr<Renderer> renderer, std::shared_ptr<World> world);

	void Update();
	void Display() const;
	void HandleMenuEnter();
	void HandleMenuExit();

	[[nodiscard]] bool InMenu() const { return in_menu_; }

private:
	void SwitchDenoiserState();
	void ChangeSelection();
	void CloseApp() const;

	template <typename T>
	bool SetValue(T& value, T min, T max)
	{
		bool value_changed = false;

		GLFWwindow* window = window_->GetGLFWWindow();

		const int left_state = glfwGetKey(window, GLFW_KEY_LEFT);
		if (left_state == GLFW_PRESS && last_left_state_ != GLFW_PRESS && value > min)
		{
			--value;
			value_changed = true;
		}

		last_left_state_ = left_state;

		const int right_state = glfwGetKey(window, GLFW_KEY_RIGHT);
		if (right_state == GLFW_PRESS && last_right_state_ != GLFW_PRESS && value < max)
		{
			++value;
			value_changed = true;
		}

		last_right_state_ = right_state;

		return value_changed;
	}

	std::shared_ptr<Window> window_ = nullptr;
	std::shared_ptr<Renderer> renderer_ = nullptr;
	std::shared_ptr<World> world_ = nullptr;
	std::unique_ptr<TextRenderer> text_renderer_ = nullptr;

	int selected_{};
	int last_left_state_{}, last_up_state_{}, last_right_state_{}, last_down_state_{};
	bool in_menu_ = true;
};