#pragma once

class Camera
{
public:
	Camera(int width, int height);

	void HandleWindowResize(int width, int height);
	void Update(GLFWwindow* window, double delta_time, bool in_menu);
	void CalculateMapping(float3& starting_point, float3& h_map, float3& v_map, float3& position) const;

	[[nodiscard]] bool Moved() const { return moved_; }
	void ResetMovedFlag() { moved_ = false; }

private:
	void Move(GLFWwindow* window, float factor);
	void Rotate(GLFWwindow* window, float factor);

	float2 prior_cursor_{ 0.0f, 0.0f }, angle_{ 0.5f * std::numbers::pi_v<float>, 0.0f };
	float3 position_{ 0.0f, 32.0f, 0.0f }, direction_{};
	float fov_{ 1.3f }, movement_speed_{ 20.0f }, rotation_speed_{ 0.02f }, aspect_ratio_;
	bool moved_{};
};