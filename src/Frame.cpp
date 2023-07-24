#include "Frame.hpp"
#include "Utils.hpp"

Frame::Frame(const int width, const int height) :
	width_(width), height_(height), frame_size_(4ull * sizeof(float) * width * height)
{
    glViewport(0, 0, width, height);

	CreateBuffer();
	CreateTexture();
}

Frame::~Frame()
{
	DeleteTexture();
	DeleteBuffer();
}

void Frame::Recreate(const int width, const int height)
{
	width_ = width;
	height_ = height;
	frame_size_ = 4ull * sizeof(float) * width * height;
	glViewport(0, 0, width, height);

	DeleteTexture();
	DeleteBuffer();
	CreateBuffer();
	CreateTexture();
}

void Frame::CreateBuffer()
{
	glGenBuffers(1, &buffer_id_);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id_);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<long long>(frame_size_), nullptr, GL_DYNAMIC_DRAW);

	CCE(cudaGraphicsGLRegisterBuffer(&cuda_resource_, buffer_id_, cudaGraphicsRegisterFlagsNone));
}

void Frame::CreateTexture()
{
	glGenTextures(1, &texture_id_);
	glBindTexture(GL_TEXTURE_2D, texture_id_);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0, GL_RGBA, GL_FLOAT, nullptr);
}

void Frame::DeleteBuffer() const
{
    CCE(cudaGraphicsUnregisterResource(cuda_resource_));

    glDeleteBuffers(1, &buffer_id_);
}

void Frame::DeleteTexture() const
{
	glDeleteTextures(1, &texture_id_);
}

void Frame::MapMemory()
{
	CCE(cudaGraphicsMapResources(1, &cuda_resource_));
	CCE(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&frame_pointer_), &frame_size_, cuda_resource_));
}

void Frame::UnmapMemory()
{
	CCE(cudaGraphicsUnmapResources(1, &cuda_resource_));
}

void Frame::Display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	glBindTexture(GL_TEXTURE_2D, texture_id_);
}
