#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/Bitmap.h"

namespace EmptySource {

	template<typename T>
	struct CubeFaceTextures {
		Bitmap<T> Right;
		Bitmap<T> Left;
		Bitmap<T> Top;
		Bitmap<T> Bottom;
		Bitmap<T> Back;
		Bitmap<T> Front;

		inline bool CheckDimensions(const int Width) const {
			if (Right.GetHeight() != Width || Right.GetWidth() != Width) return false;
			if (Left.GetHeight() != Width || Left.GetWidth() != Width) return false;
			if (Top.GetHeight() != Width || Top.GetWidth() != Width) return false;
			if (Bottom.GetHeight() != Width || Bottom.GetWidth() != Width) return false;
			if (Back.GetHeight() != Width || Back.GetWidth() != Width) return false;
			return true;
		}
	};

	template struct CubeFaceTextures<UCharRGB>;
	template struct CubeFaceTextures<FloatRGB>;

	class Texture : public std::enable_shared_from_this<Texture>{
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
		virtual void GenerateMipMaps() = 0;

		virtual unsigned int GetMipMapCount() const = 0;
		
		virtual inline EColorFormat GetColorFormat() const = 0;

		virtual inline EFilterMode GetFilterMode() const = 0;

		virtual inline IntVector3 GetSize() const = 0;

		virtual inline ETextureDimension GetDimension() const = 0;
		
		virtual inline ESamplerAddressMode GetSamplerAddressMode() const = 0;

	protected:

		virtual void SetFilterMode(const EFilterMode& Mode) = 0;

		virtual void SetSamplerAddressMode(const ESamplerAddressMode& Mode) = 0;

	};

	typedef std::shared_ptr<Texture> TexturePtr;

	typedef std::shared_ptr<class Texture2D> Texture2DPtr;

	class Texture2D : public Texture {
	public:

		virtual inline float GetAspectRatio() const = 0;

		virtual inline int GetWidth() const = 0;

		virtual inline int GetHeight() const = 0;

		static Texture2DPtr Create(
			const IntVector2& Size, const EColorFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode,
			const EColorFormat InputFormat, const void* BufferData
		);

		static Texture2DPtr Create(
			const IntVector2& Size, const EColorFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode
		);
	};

	typedef std::shared_ptr<class Cubemap> CubemapPtr;
	
	class Cubemap : public Texture {
	public:

		static CubemapPtr Create(
			const unsigned int & Size,
			const EColorFormat & Format,
			const EFilterMode & Filter,
			const ESamplerAddressMode & AddressMode
		);

		virtual bool ConvertFromCube(const CubeFaceTextures<UCharRGB>& Textures, bool GenerateMipMaps) = 0;

		virtual bool ConvertFromHDRCube(const CubeFaceTextures<FloatRGB>& Textures, bool GenerateMipMaps) = 0;

		virtual bool ConvertFromEquirectangular(Texture2DPtr Equirectangular, class Material * EquirectangularToCubemapMaterial, bool GenerateMipMaps) = 0;

		virtual bool ConvertFromHDREquirectangular(Texture2DPtr Equirectangular, class Material * EquirectangularToCubemapMaterial, bool GenerateMipMaps) = 0;

	};

}