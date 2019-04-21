
#include "../include/Texture.h"

GLuint Texture::GetColorFormat(const Graphics::ColorFormat & CF) {
	GLuint GLColorFormat = 0;
	switch (CF) {
		case Graphics::CF_Red:
			GLColorFormat = GL_RED; break;
		case Graphics::CF_RG:
			GLColorFormat = GL_RG; break;
		case Graphics::CF_RGB:
			GLColorFormat = GL_RGB; break;
		case Graphics::CF_RGBA:
			GLColorFormat = GL_RGBA; break;
		case Graphics::CF_RG16F:
			GLColorFormat = GL_RG16F; break;
		case Graphics::CF_RGB16F:
			GLColorFormat = GL_RGB16F; break;
		case Graphics::CF_RGBA16F:
			GLColorFormat = GL_RGBA16F; break;
		case Graphics::CF_RGBA32F:
			GLColorFormat = GL_RGBA32F; break;
		case Graphics::CF_RGB32F:
			GLColorFormat = GL_RGB32F; break;
		default:
			Debug::Log(Debug::LogWarning, L"Color not implemented, using RGBA");
			GLColorFormat = GL_RGBA; break;
	}

	return GLColorFormat;
}

Texture::Texture() {
	TextureObject = 0;
	FilterMode = Graphics::FM_MinMagLinear;
	AddressMode = Graphics::AM_Border;
	ColorFormat = Graphics::CF_RGB;
}

void Texture::Deuse() {
	glBindTexture(GL_TEXTURE_2D, 0);
}
