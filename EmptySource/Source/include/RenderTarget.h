#pragma once

#include "../include/Texture.h"
#include "../include/Math/MathUtility.h"
#include "../include/Math/IntVector2.h"
#include "../include/Math/Box2D.h"

namespace EmptySource {

	struct RenderTarget {
	private:
		unsigned int FramebufferObject;

		const Texture* TextureColor0Target;
		unsigned int RenderbufferObject;

		//* Render dimesions
		IntVector2 Resolution;

	public:
		//* Constructor
		RenderTarget();

		//* Get Dimension of the texture
		IntVector2 GetDimension() const;

		//* All future texture functions will modify this texture
		void PrepareTexture(const struct Texture2D * Texture, const int& Lod = 0, const int& TextureAttachment = 0);
		//* All future texture functions will modify this texture
		void PrepareTexture(const struct Cubemap * Texture, const int& TexturePos, const int& Lod = 0, const int& TextureAttachment = 0);

		//* Set resolution of render
		void Resize(const int & x, const int & y);

		//* Set up framebuffer and renderbuffer
		void SetUpBuffers();

		//* Checks the framebuffer status
		bool CheckStatus() const;

		//* Use the texture
		void Use() const;

		//* Clears the renderbuffer
		void Clear() const;

		//* Check if texture is valid
		bool IsValid() const;

		//* 
		void Delete();
	};

}