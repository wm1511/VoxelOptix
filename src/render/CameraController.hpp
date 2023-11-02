#pragma once
#include "Camera.hpp"

class CameraController
{
public:
	void UpdateCamera(GLFWwindow* window, double delta_time, bool in_menu);
	void CalculateMapping(float aspect_ratio, float3& starting_point, float3& h_map, float3& v_map, float3& position) const;

	[[nodiscard]] float3 GetCameraPosition() const { return camera_.GetPosition(); }

private:
	Camera camera_;

	float2 prior_cursor_{ 0.0f, 0.0f };
	float movement_speed_{ 20.0f }, rotation_speed_{ 0.02f }, fov_{ 1.3f };
};