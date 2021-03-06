#pragma once

namespace ESource {

	class OpenGLTexture2D : public Texture2D {
	public:
		OpenGLTexture2D(
			const IntVector2& Size, const EPixelFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode
		);

		OpenGLTexture2D(
			const IntVector2& Size, const EPixelFormat ColorFormat,
			const EFilterMode& FilterMode, const ESamplerAddressMode& AddressMode,
			const EPixelFormat InputFormat, const void* BufferData
		);

		virtual ~OpenGLTexture2D() override;

		//* Use the texture
		virtual void Bind() const override;

		//* Deuse the texture
		virtual void Unbind() const override;

		//* Returns the GL Object of this texture
		virtual void * GetNativeTexture() const override { return (void *)(unsigned long long)TextureObject; };

		//* Generate MipMaps using Hardware
		virtual void GenerateMipMaps(const EFilterMode & FilterMode, uint32_t Levels) override;

		//* Check if texture is valid
		virtual bool IsValid() const override;

	private:
		bool bValid;

		uint32_t TextureObject;

		virtual void SetFilterMode(const EFilterMode& Mode, bool MipMaps = false) override;

		virtual void SetSamplerAddressMode(const ESamplerAddressMode& Mode) override;

	};

	class OpenGLCubemap : public Cubemap {
	public:
		OpenGLCubemap(
			const int & WidthSize,
			const EPixelFormat & Format,
			const EFilterMode & Filter,
			const ESamplerAddressMode & SamplerAddress
		);

		virtual ~OpenGLCubemap() override;

		//* Use the texture
		virtual void Bind() const override;

		//* Deuse the texture
		virtual void Unbind() const override;

		//* Returns the GL Object of this texture
		virtual void * GetNativeTexture() const override { return (void *)(unsigned long long)TextureObject; };

		//* Generate MipMaps using Hardware
		virtual void GenerateMipMaps(const EFilterMode & FilterMode, uint32_t Levels) override;

		//* Check if texture is valid
		virtual bool IsValid() const override;

	private:
		bool bValid;

		uint32_t TextureObject;

		virtual void SetSamplerAddressMode(const ESamplerAddressMode& Mode) override;

		virtual void SetFilterMode(const EFilterMode& Mode, bool MipMaps = false) override;

	};
}