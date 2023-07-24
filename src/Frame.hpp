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
	void MapMemory();
	void UnmapMemory();
	void Display();
	[[nodiscard]] float* GetMemory() const { return frame_pointer_; }
	[[nodiscard]] size_t GetSize() const { return frame_size_; }

private:
	void CreateBuffer();
	void CreateTexture();
	void DeleteBuffer() const;
	void DeleteTexture() const;

	int width_ = 0, height_ = 0;
	size_t frame_size_ = 0;
	float* frame_pointer_ = nullptr;
	unsigned buffer_id_ = 0, texture_id_ = 0;
	cudaGraphicsResource_t cuda_resource_ = nullptr;
};
