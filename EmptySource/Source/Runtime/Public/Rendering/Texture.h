#pragma once

#include "Rendering/RenderingDefinitions.h"

namespace EmptySource {

	struct Texture {
	protected:
		//* Pixel Buffer GL Object
		unsigned int TextureObject;

		EFilterMode FilterMode;
		ESamplerAddressMode AddressMode;
		EColorFormat ColorFormat;

		bool bLods;

	public:
		//* Default Constructor
		Texture();

		//* Use the texture
		virtual void Use() const = 0;

		//* Deuse the texture
		virtual void Deuse() const = 0;

		//* Check if texture is valid
		virtual bool IsValid() const = 0;

		//* Returns the GL Object of this texture
		unsigned int GetTextureObject() const { return TextureObject; };

		//* Generate MipMaps using Hardware
		virtual void GenerateMipMaps() = 0;

		EFilterMode GetFilterMode() { return FilterMode; };
		ESamplerAddressMode GetSamplerAddressMode() { return AddressMode; };

		virtual void SetFilterMode(const EFilterMode& Mode) = 0;
		virtual void SetSamplerAddressMode(const ESamplerAddressMode& Mode) = 0;

		virtual void Delete() = 0;

		static unsigned int GetColorFormat(const EColorFormat & CF);
		static unsigned int GetColorFormatInput(const EColorFormat & CF);
		static unsigned int GetInputType(const EColorFormat & CF);
	};

}