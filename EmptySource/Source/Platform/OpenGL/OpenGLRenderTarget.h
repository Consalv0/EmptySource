#pragma once

#include "Rendering/RenderTarget.h"

namespace EmptySource {

	static unsigned int GetOpenGLCubemapFace(const ECubemapFace & CF);

	class OpenGLRenderTarget : public RenderTarget {
	public:
		OpenGLRenderTarget();

		virtual ~OpenGLRenderTarget() override;

		//* Use the texture
		virtual void Bind() const override;

		//* Deuse the texture
		virtual void Unbind() const override;

		//* Checks the framebuffer status
		virtual bool CheckStatus() const override;

		//* Returns the GL Object of this texture
		virtual void * GetNativeObject() const override;

		virtual TexturePtr GetBindedTexture() const override;

		//* All future functions will modify this texture
		virtual void BindTexture2D(const TexturePtr & Texture, int Lod = 0, int TextureAttachment = 0) override;

		//* All future functions will modify this texture
		virtual void BindCubemapFace(const TexturePtr & Texture, ECubemapFace Face, int Lod = 0, int TextureAttachment = 0) override;

		virtual void ReleaseTexture() override;

		//* Resize the render buffer and texture
		virtual void Resize(const IntVector3 & NewSize) override;

		//* Clear the render buffer
		virtual void Clear() const override;

		//* Generate MipMaps using Hardware
		virtual void GenerateTextureMipMaps() override;

		//* Check if texture is valid
		virtual bool IsValid() const override;

		virtual inline IntVector3 GetSize() const override { return Size; }

		virtual inline ETextureDimension GetDimension() const override { return Dimension; };

	private:

		IntVector3 Size;

		ETextureDimension Dimension;

		TexturePtr RenderingTexture;

		unsigned int FramebufferObject;

		unsigned int RenderbufferObject;
	};

}

