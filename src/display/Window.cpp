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

void Window::SetTitle(std::string name)
{
	window_name_ = std::move(name);
	glfwSetWindowTitle(window_, window_name_.c_str());
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
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef _DEBUG
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	glfwSetErrorCallback(ErrorCallback);

	window_ = glfwCreateWindow(width_, height_, window_name_.c_str(), nullptr, nullptr);
	glfwMakeContextCurrent(window_);

	if (!window_)
		throw std::exception("Failed to create GLFW window");

	if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
		throw std::exception("Failed to load GL");

	glfwSwapInterval(1);

	glfwSetWindowUserPointer(window_, this);
	glfwSetWindowSizeCallback(window_, FramebufferResizeCallback);
}