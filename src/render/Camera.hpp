#pragma once

class Camera
{
public:
	void Move(float3 delta);
	void Rotate(float2 delta);

	[[nodiscard]] float3 GetPosition() const { return position_; }
	[[nodiscard]] float3 GetDirection() const { return direction_; }

private:
	float2 angle_{ 0.5f * std::numbers::pi_v<float>, 0.0f };
	float3 position_{ 0.0f, 2.0f, 0.0f }, direction_{0.0f, 0.0f, 1.0f};
};