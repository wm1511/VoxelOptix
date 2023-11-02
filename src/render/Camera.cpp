#include "Camera.hpp"

void Camera::Move(const float3 delta)
{
	constexpr float3 up = {0.0f, 1.0f, 0.0f};
	const float3 right = normalize(cross(direction_, up));

	position_ += delta.x * right + delta.y * up + delta.z * direction_;
}

void Camera::Rotate(const float2 delta)
{
	if (length(delta) > 0.0f)
	{
		angle_ += delta;
		angle_.y = clamp(angle_.y, -1.5f, 1.5f);
		angle_.x = fmodf(angle_.x, 2.0f * std::numbers::pi_v<float>);
	}

	direction_ = normalize(make_float3(cos(angle_.x) * cos(angle_.y), sin(angle_.y), sin(angle_.x) * cos(angle_.y)));
}
