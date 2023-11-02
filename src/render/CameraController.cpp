#include "CameraController.hpp"

void CameraController::UpdateCamera(GLFWwindow* window, const double delta_time, const bool in_menu)
{
	if (!in_menu)
	{
		double current_cursor_x, current_cursor_y;
		glfwGetCursorPos(window, &current_cursor_x, &current_cursor_y);

		const float2 rotation_delta = {static_cast<float>(current_cursor_x) - prior_cursor_.x, static_cast<float>(current_cursor_y) - prior_cursor_.y};
		prior_cursor_ = make_float2(static_cast<float>(current_cursor_x), static_cast<float>(current_cursor_y));

		float3 position_delta = make_float3(0.0f);

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			position_delta.z -= 1.0f;
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			position_delta.z += 1.0f;
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			position_delta.x += 1.0f;
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			position_delta.x -= 1.0f;
		if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
			position_delta.y += 1.0f;
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
			position_delta.y -= 1.0f;

		camera_.Rotate(rotation_speed_ * static_cast<float>(delta_time) * rotation_delta);
		camera_.Move(movement_speed_ * static_cast<float>(delta_time) * position_delta);
	}
}

void CameraController::CalculateMapping(const float aspect_ratio, float3& starting_point, float3& h_map, float3& v_map, float3& position) const
{
	constexpr float3 up = {0.0f, 1.0f, 0.0f};
	position = camera_.GetPosition();
	const float3 direction = camera_.GetDirection();

	const float viewport_height = 2.0f * tanf(fov_ * 0.5f);
	const float viewport_width = viewport_height * aspect_ratio;
	const float3 u = normalize(cross(up, direction));
	const float3 v = cross(direction, u);

	h_map = viewport_width * u;
	v_map = viewport_height * v;
	starting_point = position - h_map * 0.5f - v_map * 0.5f - direction;
}