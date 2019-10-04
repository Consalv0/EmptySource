#pragma once

#include "Rendering/RenderTarget.h"

namespace ESource {

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

		virtual Texture * GetBindedTexture(int Index) const override;

		//* All future functions will modify this texture
		virtual void BindDepthTexture2D(Texture2D * Texture, const IntVector2 & Size, int Lod = 0, int TextureAttachment = 0) override;

		//* All future functions will modify this texture
		virtual void BindTexture2D(Texture2D * Texture, const IntVector2 & Size, int Lod = 0, int TextureAttachment = 0) override;

		//* All future functions will modify this texture
		virtual void BindCubemapFace(Cubemap * Texture, const int & Size, ECubemapFace Face, int Lod = 0, int TextureAttachment = 0) override;

		virtual void ReleaseTextures() override;

		//* Clear the render buffer
		virtual void Clear() const override;

		//* Check if texture is valid
		virtual bool IsValid() const override;

		virtual inline IntVector3 GetSize() const override { return Size; }

		virtual inline ETextureDimension GetDimension() const override { return Dimension; };

	private:

		IntVector3 Size;

		ETextureDimension Dimension;

		TArray<Texture *> RenderingTextures;

		unsigned int FramebufferObject;

		unsigned int RenderbufferObject;
	};

}

