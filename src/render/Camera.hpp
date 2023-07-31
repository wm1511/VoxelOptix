#pragma once

class Camera
{
public:
	Camera(int width, int height, float fov, float movement_speed, float rotation_speed);

	void Reconfigure(float fov, float movement_speed, float rotation_speed);
	void HandleWindowResize(int width, int height);
	void Update(GLFWwindow* window, double delta_time);

	[[nodiscard]] float3 GetPosition() const { return position_; }
	[[nodiscard]] float3 GetStartingPoint() const { return starting_point_; }
	[[nodiscard]] float3 GetHorizontalMap() const { return horizontal_map_; }
	[[nodiscard]] float3 GetVerticalMap() const { return vertical_map_; }

private:
	void Move(GLFWwindow* window, float factor);
	void Rotate(GLFWwindow* window, float factor);

	float2 prior_cursor_{ 0.0f, 0.0f }, angle_{0.0f, 0.0f};
	float3 position_{ 0.0f, 0.0f, 4.0f }, direction_{ 0.0f, 0.0f, -1.0f };
	float3 starting_point_{}, horizontal_map_{}, vertical_map_{}, u_{}, v_{};
	float fov_, movement_speed_, rotation_speed_, aspect_ratio_;
};