#include "Camera.hpp"

Camera::Camera(const int width, const int height, const float fov, const float movement_speed, const float rotation_speed) :
	fov_(fov),
	movement_speed_(movement_speed),
	rotation_speed_(rotation_speed),
	aspect_ratio_(width > 0 && height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f)
{
}

void Camera::Reconfigure(const float fov, const float movement_speed, const float rotation_speed)
{
	fov_ = fov;
	movement_speed_ = movement_speed;
	rotation_speed_ = rotation_speed;
}

void Camera::Move(GLFWwindow* window, const float factor, const float3 direction)
{
	constexpr float3 up = {0.0f, 1.0f, 0.0f};
	const float3 right = normalize(cross(direction, up));

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		position_ -= direction * factor;
	}
	else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		position_ += direction * factor;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		position_ += right * factor;
	}
	else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		position_ -= right * factor;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		position_ += up * factor;
	}
	else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		position_ -= up * factor;
	}
}

void Camera::Rotate(GLFWwindow* window, const float factor)
{
	double current_cursor_x, current_cursor_y;
	glfwGetCursorPos(window, &current_cursor_x, &current_cursor_y);

	const float2 delta = {static_cast<float>(current_cursor_x) - prior_cursor_.x, static_cast<float>(current_cursor_y) - prior_cursor_.y};
	prior_cursor_ = make_float2(static_cast<float>(current_cursor_x), static_cast<float>(current_cursor_y));

	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		angle_ += delta * factor;
		angle_.y = clamp(angle_.y, -1.5f, 1.5f);
		angle_.x = fmodf(angle_.x, 2.0f * std::numbers::pi_v<float>);
	}
}

void Camera::HandleWindowResize(const int width, const int height)
{
	aspect_ratio_ = width > 0 && height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
}

void Camera::Update(GLFWwindow* window, const double delta_time, const bool in_menu)
{
	constexpr float3 up = {0.0f, 1.0f, 0.0f};
	const float3 direction = normalize(make_float3(cos(angle_.x) * cos(angle_.y), sin(angle_.y), sin(angle_.x) * cos(angle_.y)));

	if (!in_menu)
	{
		Move(window, movement_speed_ * static_cast<float>(delta_time), direction);
		Rotate(window, rotation_speed_ * static_cast<float>(delta_time));
	}

	const float viewport_height = 2.0f * tanf(fov_ * 0.5f);
	const float viewport_width = viewport_height * aspect_ratio_;

	u_ = normalize(cross(up, direction));
	v_ = cross(direction, u_);
	horizontal_map_ = viewport_width * u_;
	vertical_map_ = viewport_height * v_;
	starting_point_ = position_ - horizontal_map_ * 0.5f - vertical_map_ * 0.5f - direction;
}
