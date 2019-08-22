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

		virtual inline int GetWidth() const override { return Size.x; }

		virtual inline int GetHeight() const override { return Size.y; }

		virtual inline float GetAspectRatio() const override { return (float)Size.x / (float)Size.y; };

		virtual inline IntVector3 GetSize() const override { return Size; }

		virtual inline ETextureDimension GetDimension() const override { return ETextureDimension::Texture2D; };

		virtual unsigned int GetMipMapCount() const override { return MipMapCount; };

		virtual inline EFilterMode GetFilterMode() const override { return FilterMode; };

		virtual inline EColorFormat GetColorFormat() const override { return ColorFormat; }

		virtual inline ESamplerAddressMode GetSamplerAddressMode() const override { return AddressMode; };

	private:
		IntVector2 Size;

		bool bValid;

		unsigned int TextureObject;

		EFilterMode FilterMode;

		ESamplerAddressMode AddressMode;
		
		EColorFormat ColorFormat;

		unsigned int MipMapCount;

		virtual void SetFilterMode(const EFilterMode& Mode) override;

		virtual void SetSamplerAddressMode(const ESamplerAddressMode& Mode) override;

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

		virtual inline IntVector3 GetSize() const override { return IntVector2(Size); }

		virtual inline ETextureDimension GetDimension() const override { return ETextureDimension::Cubemap; };

		virtual unsigned int GetMipMapCount() const override { return MipMapCount; };

		virtual inline EFilterMode GetFilterMode() const override { return FilterMode; };

		virtual inline ESamplerAddressMode GetSamplerAddressMode() const override { return AddressMode; };

		virtual inline EColorFormat GetColorFormat() const override { return ColorFormat; }

		virtual bool ConvertFromCube(const CubeFaceTextures<UCharRGB>& Textures, bool GenerateMipMaps) override;

		virtual bool ConvertFromHDRCube(const CubeFaceTextures<FloatRGB>& Textures, bool GenerateMipMaps) override;

		virtual bool ConvertFromEquirectangular(Texture2DPtr Equirectangular, Material * EquirectangularToCubemapMaterial, bool GenerateMipMaps) override;

		virtual bool ConvertFromHDREquirectangular(Texture2DPtr Equirectangular, Material * EquirectangularToCubemapMaterial, bool GenerateMipMaps) override;

	private:

		unsigned int Size;

		bool bValid;

		unsigned int TextureObject;

		EFilterMode FilterMode;

		ESamplerAddressMode AddressMode;

		EColorFormat ColorFormat;

		unsigned int MipMapCount;

		virtual void SetSamplerAddressMode(const ESamplerAddressMode& Mode) override;

		virtual void SetFilterMode(const EFilterMode& Mode) override;

	};
}