#pragma once

#include "..\include\Graphics.h"
#include "..\include\Text.h"
#include "..\include\Math\IntVector3.h"

struct Texture {
protected:
	//* Pixel Buffer GL Object
	unsigned int TextureObject;

	Graphics::FilterMode FilterMode;
	Graphics::AddressMode AddressMode;
	Graphics::ColorFormat ColorFormat;

public:
	//* Default Constructor
	Texture();

	//* Use the texture
	virtual void Use() const = 0;

	//* Check if texture is valid
	virtual bool IsValid() const = 0;

	//* Returns the GL Object of this texture
	unsigned int GetTextureObject() const { return TextureObject; };

	Graphics::FilterMode GetFilterMode() { return FilterMode; };
	Graphics::AddressMode GetAddressMode() { return AddressMode; };

	virtual void SetFilterMode(const Graphics::FilterMode& Mode) = 0;
	virtual void SetAddressMode(const Graphics::AddressMode& Mode) = 0;

	virtual void Delete() = 0;
};

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
