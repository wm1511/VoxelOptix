#pragma once

class Camera
{
public:
	Camera(int width, int height);

	void HandleWindowResize(int width, int height);
	void Update(GLFWwindow* window, double delta_time, bool in_menu);

	[[nodiscard]] float3 GetPosition() const { return position_; }
	[[nodiscard]] float3 GetStartingPoint() const { return starting_point_; }
	[[nodiscard]] float3 GetHorizontalMap() const { return horizontal_map_; }
	[[nodiscard]] float3 GetVerticalMap() const { return vertical_map_; }

private:
	void Move(GLFWwindow* window, float factor, float3 direction);
	void Rotate(GLFWwindow* window, float factor);

	float2 prior_cursor_{ 0.0f, 0.0f }, angle_{ 0.5f * std::numbers::pi_v<float>, 0.0f };
	float3 position_{ 0.0f, 32.0f, 0.0f };
	float3 starting_point_{}, horizontal_map_{}, vertical_map_{}, u_{}, v_{};
	float fov_{ 1.3f }, movement_speed_{ 20.0f }, rotation_speed_{ 0.02f }, aspect_ratio_;
	};