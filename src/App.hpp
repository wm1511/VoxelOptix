#pragma once
#include "Frame.hpp"
#include "Window.hpp"

class App
{
public:
	App();
	~App() = default;

	App(const App&) = delete;
	App(App&&) = delete;
	App& operator= (const App&) = delete;
	App& operator= (App&&) = delete;

	void Run();

private:
	void OnUpdate();
	void OnResize() const;

	double delta_time_ = 0.0, last_frame_ = 0.0;	
	std::unique_ptr<Window> window_ = nullptr;
	std::unique_ptr<Frame> frame_ = nullptr;
};
