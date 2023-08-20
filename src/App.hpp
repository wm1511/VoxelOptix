#pragma once
#include "display/Window.hpp"
#include "display/Frame.hpp"
#include "display/Menu.hpp"
#include "render/Renderer.hpp"
#include "world/World.hpp"

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
	void OnceASecond() const;
	void WorkInBackground() const;

	bool worker_running_{ true };
	double delta_time_ = 0.0, last_frame_ = 0.0;
	std::shared_ptr<Window> window_ = nullptr;
	std::unique_ptr<Frame> frame_ = nullptr;
	std::shared_ptr<Camera> camera_ = nullptr;
	std::shared_ptr<World> world_ = nullptr;
	std::shared_ptr<Renderer> renderer_ = nullptr;
	std::unique_ptr<Menu> menu_ = nullptr;
};
