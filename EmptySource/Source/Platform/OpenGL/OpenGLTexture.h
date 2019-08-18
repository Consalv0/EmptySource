#pragma once

namespace EmptySource {

	static unsigned int GetOpenGLTextureColorFormat(const EColorFormat & CF);
	
	static unsigned int GetOpenGLTextureColorFormatInput(const EColorFormat & CF);

	static unsigned int GetOpenGLTextureInputType(const EColorFormat & CF);

	class OpenGLTexture2D : public Texture2D {
	public:
		OpenGLTexture2D(
			const IntVector2& Size, const EColorFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode
		);

		OpenGLTexture2D(
			const IntVector2& Size, const EColorFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode,
			const EColorFormat InputFormat, const void* BufferData
		);

		virtual ~OpenGLTexture2D() override;

		//* Use the texture
		virtual void Bind() const override;

		//* Deuse the texture
		virtual void Unbind() const override;

		//* Returns the GL Object of this texture
		virtual void * GetTextureObject() const override { return (void *)(unsigned long long)TextureObject; };

		//* Generate MipMaps using Hardware
		virtual void GenerateMipMaps() override;

		//* Check if texture is valid
		virtual bool IsValid() const override;

		virtual inline int GetWidth() const override { return Dimensions.x; }

		virtual inline int GetHeight() const override { return Dimensions.y; }

		virtual inline float GetAspectRatio() const override { return (float)Dimensions.x / (float)Dimensions.y; };

		virtual inline IntVector3 GetDimensions() const override { return Dimensions; };

		virtual unsigned int GetMipMapCount() const override { return MipMapCount; };

		virtual inline EFilterMode GetFilterMode() const override { return FilterMode; };

		virtual void SetFilterMode(const EFilterMode& Mode) override;

		virtual inline ESamplerAddressMode GetSamplerAddressMode() const override { return AddressMode; };

		virtual inline EColorFormat GetColorFormat() const override { return ColorFormat; }

		virtual void SetSamplerAddressMode(const ESamplerAddressMode& Mode) override;

	private:

		IntVector2 Dimensions;

		bool bValid;

		unsigned int TextureObject;

		EFilterMode FilterMode;

		ESamplerAddressMode AddressMode;
		
		EColorFormat ColorFormat;

		unsigned int MipMapCount;

	};

	class OpenGLCubemap : public Cubemap {
	public:
		OpenGLCubemap(
			const int & WidthSize,
			const EColorFormat & Format,
			const EFilterMode & Filter,
			const ESamplerAddressMode & SamplerAddress
		);

		virtual ~OpenGLCubemap() override;

		//* Use the texture
		virtual void Bind() const override;

		//* Deuse the texture
		virtual void Unbind() const override;

		//* Returns the GL Object of this texture
		virtual void * GetTextureObject() const override { return (void *)(unsigned long long)TextureObject; };

		//* Generate MipMaps using Hardware
		virtual void GenerateMipMaps() override;

		//* Check if texture is valid
		virtual bool IsValid() const override;

		virtual inline unsigned int GetSize() const override { return Size; }

		virtual inline IntVector3 GetDimensions() const override { return IntVector2(Size); };

		virtual unsigned int GetMipMapCount() const override { return MipMapCount; };

		virtual inline EFilterMode GetFilterMode() const override { return FilterMode; };

		virtual void SetFilterMode(const EFilterMode& Mode) override;

		virtual inline ESamplerAddressMode GetSamplerAddressMode() const override { return AddressMode; };

		virtual inline EColorFormat GetColorFormat() const override { return ColorFormat; }

		virtual void SetSamplerAddressMode(const ESamplerAddressMode& Mode) override;

		static bool ConvertFromCube(CubemapPtr & Cube, const TextureData<UCharRGB>& Textures);

		static bool ConvertFromHDRCube(CubemapPtr & Cube, const TextureData<FloatRGB>& Textures);

		static bool ConvertFromEquirectangular(CubemapPtr & Cube, Texture2DPtr Equirectangular, Material * EquirectangularToCubemapMaterial);

		static bool ConvertFromHDREquirectangular(CubemapPtr & Cube, Texture2DPtr Equirectangular, Material * EquirectangularToCubemapMaterial);

	private:

		unsigned int Size;

		bool bValid;

		unsigned int TextureObject;

		EFilterMode FilterMode;

		ESamplerAddressMode AddressMode;

		EColorFormat ColorFormat;

		unsigned int MipMapCount;

	};
}