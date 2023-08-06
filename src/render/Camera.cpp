#include "Camera.hpp"

Camera::Camera(const int width, const int height) :
	aspect_ratio_(width > 0 && height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f)
{
}

void Camera::Move(GLFWwindow* window, const float factor)
{
	constexpr float3 up = {0.0f, 1.0f, 0.0f};
	const float3 right = normalize(cross(direction_, up));

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		position_ -= direction_ * factor;
		moved_ = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		position_ += direction_ * factor;
		moved_ = true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		position_ += right * factor;
		moved_ = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		position_ -= right * factor;
		moved_ = true;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		position_ += up * factor;
		moved_ = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		position_ -= up * factor;
		moved_ = true;
	}
}

void Camera::Rotate(GLFWwindow* window, const float factor)
{
	double current_cursor_x, current_cursor_y;
	glfwGetCursorPos(window, &current_cursor_x, &current_cursor_y);

	const float2 delta = {static_cast<float>(current_cursor_x) - prior_cursor_.x, static_cast<float>(current_cursor_y) - prior_cursor_.y};
	prior_cursor_ = make_float2(static_cast<float>(current_cursor_x), static_cast<float>(current_cursor_y));

	if (length(delta) > 0.0f)
	{
		angle_ += delta * factor;
		angle_.y = clamp(angle_.y, -1.5f, 1.5f);
		angle_.x = fmodf(angle_.x, 2.0f * std::numbers::pi_v<float>);
		moved_ = true;
	}
}

void Camera::HandleWindowResize(const int width, const int height)
{
	aspect_ratio_ = width > 0 && height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
}

void Camera::Update(GLFWwindow* window, const double delta_time, const bool in_menu)
{
	direction_ = normalize(make_float3(cos(angle_.x) * cos(angle_.y), sin(angle_.y), sin(angle_.x) * cos(angle_.y)));

	if (!in_menu)
	{
		Move(window, movement_speed_ * static_cast<float>(delta_time));
		Rotate(window, rotation_speed_ * static_cast<float>(delta_time));
	}
}

void Camera::CalculateMapping(float3& starting_point, float3& h_map, float3& v_map, float3& position) const
{
	constexpr float3 up = {0.0f, 1.0f, 0.0f};

	const float viewport_height = 2.0f * tanf(fov_ * 0.5f);
	const float viewport_width = viewport_height * aspect_ratio_;
	const float3 u = normalize(cross(up, direction_));
	const float3 v = cross(direction_, u);

	h_map = viewport_width * u;
	v_map = viewport_height * v;
	starting_point = position_ - h_map * 0.5f - v_map * 0.5f - direction_;
	position = position_;
}
