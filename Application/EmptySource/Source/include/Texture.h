#pragma once

#include "../include/Core.h"
#include "../include/CoreGraphics.h"

struct Texture {
protected:
	//* Pixel Buffer GL Object
	unsigned int TextureObject;

	Graphics::FilterMode FilterMode;
	Graphics::AddressMode AddressMode;
	Graphics::ColorFormat ColorFormat;

	bool bLods;

public:
	//* Default Constructor
	Texture();

	//* Use the texture
	virtual void Use() const = 0;

	//* Use the texture
	static void Deuse();

	//* Check if texture is valid
	virtual bool IsValid() const = 0;

	//* Returns the GL Object of this texture
	unsigned int GetTextureObject() const { return TextureObject; };

	virtual void GenerateMipMaps() = 0;

	Graphics::FilterMode GetFilterMode() { return FilterMode; };
	Graphics::AddressMode GetAddressMode() { return AddressMode; };

	virtual void SetFilterMode(const Graphics::FilterMode& Mode) = 0;
	virtual void SetAddressMode(const Graphics::AddressMode& Mode) = 0;

	virtual void Delete() = 0;

	static GLuint GetColorFormat(const Graphics::ColorFormat & CF);
};
