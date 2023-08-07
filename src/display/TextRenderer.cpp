#include "TextRenderer.hpp"
#include "../misc/Exceptions.hpp"
#include "../external/NinePin.h"

TextRenderer::TextRenderer()
{
	CreateBuffers();
	CreateShader();
	LoadFont();
}

void TextRenderer::Render(float x, float y, const std::string& text, const float scale, const float3 color)
{
	// Enable blending to see frame below text
	CGE(glEnable(GL_BLEND));
	CGE(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

	// Binding shader
	CGE(glUseProgram(shader_));
	const int sampler_location = CGE(glGetUniformLocation(shader_, "textureSampler"));
	CGE(glUniform1i(sampler_location, 0));
	const int color_location = CGE(glGetUniformLocation(shader_, "textColor"));
	CGE(glUniform3f(color_location, color.x, color.y, color.z));

	// Binding vertex data
	glBindVertexArray(vao_);
	CGE(glBindBuffer(GL_ARRAY_BUFFER, vbo_));

	// Setting texture unit active
	CGE(glActiveTexture(GL_TEXTURE0));

	CenterText(x, y, text, scale);

	for (const auto c : text)
	{
		RenderCharacter(x, y, c, scale);
		x += 6.0f * pixel_size_.x * scale;
	}

	// Cleanup
	CGE(glBindBuffer(GL_ARRAY_BUFFER, 0));
	CGE(glBindVertexArray(0));
	CGE(glUseProgram(0));

	CGE(glDisable(GL_BLEND));
}

void TextRenderer::LoadFont()
{
	CGE(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));

	unsigned texture = 0, data_offset = 0;

	for (int i = 32; i < 127; i++)
	{
		CGE(glGenTextures(1, &texture));
		CGE(glBindTexture(GL_TEXTURE_2D, texture));

		CGE(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
		CGE(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

		CGE(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		CGE(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

		CGE(glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 5, 8, 0, GL_RED, GL_UNSIGNED_BYTE,
			reinterpret_cast<void*>(ninepin::bitmaps + data_offset)));

		characters_.insert(std::pair(i, texture));

		data_offset += 40;
	}
}

void TextRenderer::CenterText(float& x, float& y, const std::string& text, const float scale) const
{
	const float2 size = make_float2(static_cast<float>(text.size()) * 6.0f - 1.0f, 8.0f) * pixel_size_ * scale;

	x -= size.x * 0.5f;
	y -= size.y * 0.5f;
}

void TextRenderer::RenderCharacter(const float x, const float y, const char character, const float scale)
{
	const float vertices[] =
	{
		x,								  y + 8.0f * pixel_size_.y * scale, 0.0f, 0.0f,
		x,								  y,								0.0f, 1.0f,
		x + 5.0f * pixel_size_.x * scale, y + 8.0f * pixel_size_.y * scale, 1.0f, 0.0f,
		x + 5.0f * pixel_size_.x * scale, y,								1.0f, 1.0f
	};

	// Setting current character data
	CGE(glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof vertices, vertices));
	CGE(glBindTexture(GL_TEXTURE_2D, characters_[character]));

	// Drawing character
	CGE(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
}

void TextRenderer::CreateShader()
{
	const auto vertex_source = R"(
	    #version 330 core
	    layout(location = 0) in vec4 aPos;

	    out vec2 TexCoord;

	    void main() 
		{
			gl_Position = vec4(aPos.xy, 0.0, 1.0);
	        TexCoord = aPos.zw;
	    }
	)";

	const auto fragment_source = R"(
	    #version 330 core
		layout(location = 0) out vec4 Color;

	    in vec2 TexCoord;
	    uniform sampler2D textureSampler;
		uniform vec3 textColor;

	    void main() 
		{
			Color = vec4(textColor, texture(textureSampler, TexCoord).r);
	    }
	)";

	int success = 0;
	char log[512];

	const unsigned int vertex = CGE(glCreateShader(GL_VERTEX_SHADER));
	CGE(glShaderSource(vertex, 1, &vertex_source, nullptr));
	CGE(glCompileShader(vertex));

	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertex, 512, nullptr, log);
		throw std::exception(log);
	}

	const unsigned int fragment = CGE(glCreateShader(GL_FRAGMENT_SHADER));
	CGE(glShaderSource(fragment, 1, &fragment_source, nullptr));
	CGE(glCompileShader(fragment));

	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragment, 512, nullptr, log);
		throw std::exception(log);
	}

	shader_ = CGE(glCreateProgram());
	CGE(glAttachShader(shader_, vertex));
	CGE(glAttachShader(shader_, fragment));
	CGE(glLinkProgram(shader_));

	glGetProgramiv(shader_, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(shader_, 512, nullptr, log);
		throw std::exception(log);
	}

	CGE(glDetachShader(shader_, vertex));
	CGE(glDetachShader(shader_, fragment));
}

void TextRenderer::CreateBuffers()
{
	CGE(glGenVertexArrays(1, &vao_));
	CGE(glBindVertexArray(vao_));

	CGE(glGenBuffers(1, &vbo_));
	CGE(glBindBuffer(GL_ARRAY_BUFFER, vbo_));
	CGE(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 16, nullptr, GL_DYNAMIC_DRAW));

	CGE(glEnableVertexAttribArray(0));
	CGE(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr));

	CGE(glBindVertexArray(0));
	CGE(glBindBuffer(GL_ARRAY_BUFFER, 0));
}
