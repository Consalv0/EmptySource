#pragma once

#include "..\include\Texture.h"

struct Texture2D : public Texture {
private:
	//* Texture dimesions
	IntVector2 Dimension;

public:
	//* Constructor
	Texture2D(
		const IntVector2& Size, const Graphics::ColorFormat ColorFormat,
		const Graphics::FilterMode& FilterMode, const Graphics::AddressMode& AddressMode
	);

	//* Get Dimension of the texture
	IntVector2 GetDimension() const;

	//* Use the texture
	void Use() const;

	//* Check if texture is valid
	bool IsValid() const;

	void SetFilterMode(const Graphics::FilterMode& Mode);

	void SetAddressMode(const Graphics::AddressMode& Mode);

	//* 
	void Delete();
};