#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/PixelMap.h"

namespace ESource {

	struct CubeFaceTextures {
		PixelMap Right;
		PixelMap Left;
		PixelMap Top;
		PixelMap Bottom;
		PixelMap Back;
		PixelMap Front;

		inline bool CheckDimensions(const int Width) const {
			if ( Right.GetHeight() != Width ||  Right.GetWidth() != Width) return false;
			if (  Left.GetHeight() != Width ||   Left.GetWidth() != Width) return false;
			if (   Top.GetHeight() != Width ||    Top.GetWidth() != Width) return false;
			if (Bottom.GetHeight() != Width || Bottom.GetWidth() != Width) return false;
			if (  Back.GetHeight() != Width ||   Back.GetWidth() != Width) return false;
			return true;
		}
	};

	class Texture {
	public:
		virtual ~Texture() = default;

		//* Use the texture
		virtual void Bind() const = 0;

		//* Deuse the texture
		virtual void Unbind() const = 0;

		//* Check if texture is valid
		virtual bool IsValid() const = 0;

		//* Returns the GL Object of this texture
		virtual void * GetTextureObject() const = 0;

		//* Generate MipMaps using Hardware
		virtual void GenerateMipMaps(const EFilterMode & FilterMode, unsigned int Levels) = 0;

	protected:
		virtual void SetFilterMode(const EFilterMode& Mode, bool MipMaps) = 0;

		virtual void SetSamplerAddressMode(const ESamplerAddressMode& Mode) = 0;
	};

	class Texture2D : public Texture {
	public:
		static Texture2D * Create(
			const IntVector2& Size, const EPixelFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode,
			const EPixelFormat InputFormat, const void* BufferData
		);

		static Texture2D * Create(
			const IntVector2& Size, const EPixelFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode
		);
	};

	class Cubemap : public Texture {
	public:
		static Cubemap * Create(
			const unsigned int & Size,
			const EPixelFormat & Format,
			const EFilterMode & Filter,
			const ESamplerAddressMode & AddressMode
		);
	};

}