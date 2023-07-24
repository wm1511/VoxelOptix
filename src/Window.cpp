#include "Window.hpp"

Window::Window(const int width, const int height, std::string name) : width_(width), height_(height), window_name_(std::move(name))
{
	Init();
}

Window::~Window()
{
	glfwDestroyWindow(window_);
	glfwTerminate();
}

void Window::FramebufferResizeCallback(GLFWwindow* window, const int width, const int height)
{
	const auto app_window = static_cast<Window*>(glfwGetWindowUserPointer(window));
	app_window->framebuffer_resized_ = true;
	app_window->width_ = width;
	app_window->height_ = height;
}

void Window::ErrorCallback(const int error, const char* description)
{
	throw std::exception(description, error);
}

void Window::Init()
{
	if (!glfwInit())
		throw std::exception("Failed to init GLFW");

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
	glfwWindowHint(GLFW_SAMPLES, 4);
#ifdef _DEBUG
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

	glfwSetErrorCallback(ErrorCallback);

	window_ = glfwCreateWindow(width_, height_, window_name_.c_str(), nullptr, nullptr);
	glfwMakeContextCurrent(window_);

	if (!window_)
		throw std::exception("Failed to create GLFW window");

	if (!gladLoadGL()) 
        throw std::exception("Failed to load GL");

	glfwSwapInterval(1);

	glfwSetWindowUserPointer(window_, this);
	glfwSetWindowSizeCallback(window_, FramebufferResizeCallback);
}