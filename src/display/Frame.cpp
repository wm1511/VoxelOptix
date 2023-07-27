#include "Frame.hpp"
#include "../misc/Exceptions.hpp"

Frame::Frame(const int width, const int height) :
	width_(width), height_(height)
{
	CCE(cudaSetDevice(0));
	Recreate(width, height);
	CreateShader();
	CreateTexturedQuad();
}

Frame::~Frame()
{
	CCE(cudaGraphicsUnregisterResource(cuda_pbo_));

	if (pbo_)
	{
		CGE(glBindBuffer(GL_ARRAY_BUFFER, 0));
		CGE(glDeleteBuffers(1, &pbo_));
	}
}

void Frame::Recreate(const int width, const int height)
{
	if (width < 64 || height < 64)
		return;

	width_ = width;
	height_ = height;

	CGE(glGenBuffers(1, &pbo_));
    CGE(glBindBuffer(GL_ARRAY_BUFFER, pbo_));
    CGE(glBufferData(GL_ARRAY_BUFFER, static_cast<long long>(sizeof(float4)) * width_ * height_, nullptr, GL_STREAM_DRAW));
    CGE(glBindBuffer(GL_ARRAY_BUFFER, 0));

	CCE(cudaGraphicsGLRegisterBuffer(&cuda_pbo_, pbo_, cudaGraphicsMapFlagsWriteDiscard));
}

void Frame::CreateTexturedQuad()
{
	unsigned vao;
	constexpr float vertices[] =
	{
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 0.0f
	};

	CGE(glGenVertexArrays(1, &vao));
	CGE(glBindVertexArray(vao));

	CGE(glGenTextures(1, &texture_));
	CGE(glBindTexture(GL_TEXTURE_2D, texture_));

	CGE(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	CGE(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	CGE(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	CGE(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

	CGE(glGenBuffers(1, &vbo_));
	CGE(glBindBuffer(GL_ARRAY_BUFFER, vbo_));
	CGE(glBufferData(GL_ARRAY_BUFFER, sizeof vertices, vertices, GL_STATIC_DRAW));
}

void Frame::CreateShader()
{
	const auto vertex_source = R"(
	    #version 330 core
	    layout(location = 0) in vec3 aPos;

	    out vec2 TexCoord;

	    void main() 
		{
			gl_Position = vec4(aPos, 1.0);
	        TexCoord = (aPos.xy + vec2(1.0)) / 2.0;
	    }
	)";

	const auto fragment_source = R"(
	    #version 330 core
		layout(location = 0) out vec3 Color;

	    in vec2 TexCoord;
	    uniform sampler2D textureSampler;

	    void main() 
		{
			Color = texture(textureSampler, TexCoord).xyz;
	    }
	)";

	const unsigned int vertex = CGE(glCreateShader(GL_VERTEX_SHADER));
	CGE(glShaderSource(vertex, 1, &vertex_source, nullptr));
	CGE(glCompileShader(vertex));

	const unsigned int fragment = CGE(glCreateShader(GL_FRAGMENT_SHADER));
	CGE(glShaderSource(fragment, 1, &fragment_source, nullptr));
	CGE(glCompileShader(fragment));

	shader_ = CGE(glCreateProgram());
	CGE(glAttachShader(shader_, vertex));
	CGE(glAttachShader(shader_, fragment));
	CGE(glLinkProgram(shader_));

	CGE(glDetachShader(shader_, vertex));
	CGE(glDetachShader(shader_, fragment));
}

float4* Frame::MapMemory()
{
	float4* frame_pointer = nullptr;
	size_t frame_size = 0;

	CCE(cudaGraphicsMapResources(1, &cuda_pbo_));
	CCE(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&frame_pointer), &frame_size, cuda_pbo_));

	return frame_pointer;
}

void Frame::UnmapMemory()
{
	CCE(cudaGraphicsUnmapResources(1, &cuda_pbo_));
}

void Frame::Display() const
{
	CGE(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	CGE(glViewport(0, 0, width_, height_));

	CGE(glClearColor(1.0f, 0.0f, 0.0f, 1.0f));
	CGE(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
	CGE(glUseProgram(shader_));

	CGE(glActiveTexture(GL_TEXTURE0));
	CGE(glBindTexture(GL_TEXTURE_2D, texture_));
	CGE(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_));

	CGE(glPixelStorei(GL_UNPACK_ALIGNMENT, 8));
	CGE(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0, GL_RGBA, GL_FLOAT, nullptr));

	CGE(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
	const int uniform_location = CGE(glGetUniformLocation(shader_, "textureSampler"));
	CGE(glUniform1i(uniform_location, 0));

	CGE(glEnableVertexAttribArray(0));
    CGE(glBindBuffer(GL_ARRAY_BUFFER, vbo_));
    CGE(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, static_cast<void*>(nullptr)));

	CGE(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
	CGE(glDisableVertexAttribArray(0));
}
