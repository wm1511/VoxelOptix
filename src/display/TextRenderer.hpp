#pragma once

class TextRenderer
{
public:
	TextRenderer();

	void Render(float x, float y, const std::string& text, float scale = 8.0f, float3 color = {1.0f, 1.0f, 1.0f});

private:
	void RenderCharacter(float x, float y, char character, float scale);
	void CreateShader();
	void CreateBuffers();
	void LoadFont();
	void CenterText(float& x, float& y, const std::string& text, float scale) const;

	unsigned vao_, vbo_, shader_;
	float2 pixel_size_{5.2083e-4f, 9.259e-4f};
	std::unordered_map<int, unsigned> characters_{};
};
