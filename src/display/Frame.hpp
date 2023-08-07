#pragma once

class Frame
{
public:
	Frame(int width, int height);
	~Frame();

	Frame(const Frame&) = delete;
	Frame(Frame&&) = delete;
	Frame& operator= (const Frame&) = delete;
	Frame& operator= (Frame&&) = delete;

	void Recreate(int width, int height);
	float4* MapMemory();
	void UnmapMemory();
	void Display() const;

private:
	void CreateTexturedQuad();
	void CreateShader();

	int width_ = 0, height_ = 0;
	unsigned vao_ = 0, vbo_ = 0, pbo_ = 0, texture_ = 0, shader_ = 0;
	cudaGraphicsResource_t cuda_pbo_ = nullptr;
};
