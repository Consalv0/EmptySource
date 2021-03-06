#pragma once

#include "Rendering/RenderTarget.h"

namespace ESource {

	static uint32_t GetOpenGLCubemapFace(const ECubemapFace & CF);

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
		virtual void BindDepthTexture2D(Texture2D * Texture, EPixelFormat Format, const IntVector2 & Size, int Lod = 0) override;

		//* All future functions will modify this texture
		virtual void BindTexture2D(Texture2D * Texture, const IntVector2 & Size, int Lod = 0, int TextureAttachment = 0) override;

		//* All future functions will modify this textures
		virtual void BindTextures2D(Texture2D ** Textures, const IntVector2 & Size, int * Lods, int * TextureAttachments, uint32_t Count) override;

		//* All future functions will modify this texture
		virtual void BindCubemapFace(Cubemap * Texture, const int & Size, ECubemapFace Face, int Lod = 0, int TextureAttachment = 0) override;

		virtual void CreateRenderDepthBuffer2D(EPixelFormat Format, const IntVector2 & Size) override;
		
		virtual void TransferBitsTo(RenderTarget * Target, bool Color, bool Stencil, bool Depth, const EFilterMode & FilterMode, const IntBox2D & From, const IntBox2D & To) override;

		virtual void ReleaseTextures() override;

		//* Clear the render buffer
		virtual void Clear() const override;

		//* Check if texture is valid
		virtual bool IsValid() const override;

		virtual inline IntVector3 GetSize() const override { return Size; }

		virtual inline IntBox2D GetViewport() const override { return IntBox2D(0, 0, Size.X, Size.Y); }

		virtual inline ETextureDimension GetDimension() const override { return Dimension; };

	private:
		bool bHasDepth;

		bool bHasStencil;

		bool bHasColor;

		IntVector3 Size;

		ETextureDimension Dimension;

		TArray<Texture *> RenderingTextures;

		uint32_t FramebufferObject;

		uint32_t RenderbufferObject;
	};

}

