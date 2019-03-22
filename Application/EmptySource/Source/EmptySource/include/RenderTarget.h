#pragma once

#include "../include/Texture.h"

struct RenderTarget {
private:
	unsigned int FramebufferObject;

	Texture* TextureColor0Target; 
	unsigned int TextureDepthTarget;

	//* Texture dimesions
	IntVector2 Dimension; 

public:
	//* Constructor
	RenderTarget();

	RenderTarget(
		const IntVector2& Size, Texture* Color0, bool bUseDepth
	);

	//* Get Dimension of the texture
	IntVector2 GetDimension() const;

	//* Use the texture
	void Use() const;

	void Clear() const;

	//* Check if texture is valid
	bool IsValid() const;

	//* 
	void Delete();
};
