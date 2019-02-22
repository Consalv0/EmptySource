#pragma once

#include "..\include\Texture.h"

struct Texture3D : public Texture {
private:
	//* Texture dimesions
	IntVector3 Dimension;

public:
	//* Constructor
	Texture3D(
		const IntVector3& Size, const Graphics::ColorFormat ColorFormat,
		const Graphics::FilterMode& FilterMode, const Graphics::AddressMode& AddressMode
	);

	//* Get Dimension of the texture
	IntVector3 GetDimension() const;

	//* Use the texture
	void Use() const;

	//* Check if texture is valid
	bool IsValid() const;

	void SetFilterMode(const Graphics::FilterMode& Mode);

	void SetAddressMode(const Graphics::AddressMode& Mode);

	//* 
	void Delete();
};