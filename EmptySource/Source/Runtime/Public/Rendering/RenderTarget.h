#pragma once

#include "Rendering/Texture.h"
#include "Math/MathUtility.h"
#include "Math/IntVector2.h"
#include "Math/Box2D.h"

namespace ESource {

	typedef std::shared_ptr<class RenderTarget> RenderTargetPtr;

	class RenderTarget {
	public:
		virtual ~RenderTarget() = default;

		//* Get the dimension of the texture
		virtual inline ETextureDimension GetDimension() const = 0;

		//* All future functions will modify this texture
		virtual void BindDepthTexture2D(Texture2D * Texture, const IntVector2 & Size, int Lod = 0, int TextureAttachment = 0)  = 0;

		//* All future functions will modify this texture
		virtual void BindTexture2D(Texture2D * Texture, const IntVector2 & Size, int Lod = 0, int TextureAttachment = 0) = 0;

		//* All future functions will modify this texture
		virtual void BindCubemapFace(Cubemap * Texture, const int & Size, ECubemapFace Face, int Lod = 0, int TextureAttachment = 0) = 0;

		//* Returns empty if no texture
		virtual Texture * GetBindedTexture() const = 0;

		//* Get the renderbuffer object
		virtual void * GetNativeObject() const = 0;

		//* Set size of render
		virtual void Resize(const IntVector3 & NewSize) = 0;

		//* Get size of render
		virtual IntVector3 GetSize() const = 0;

		//* Checks the framebuffer status
		virtual bool CheckStatus() const = 0;

		//* Bind the render target
		virtual void Bind() const = 0;

		//* Unbind the render target
		virtual void Unbind() const = 0;

		//* Clears the texture binded
		virtual void Clear() const = 0;

		//* Release the texture binded
		virtual void ReleaseTexture() = 0;

		//* Check if render target is valid
		virtual bool IsValid() const = 0;

		static RenderTargetPtr Create();
	};

}