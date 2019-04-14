
#include "../include/Texture.h"

GLuint Texture::GetColorFormat(const Graphics::ColorFormat & CF) {
	GLuint GLColorFormat = 0;
	switch (CF) {
		case Graphics::CF_Red:
			GLColorFormat = GL_RED; break;
		case Graphics::CF_RGB:
			GLColorFormat = GL_RGB; break;
		case Graphics::CF_RGBA:
			GLColorFormat = GL_RGBA; break;
		case Graphics::CF_RGBA16F:
			GLColorFormat = GL_RGBA16F; break;
		case Graphics::CF_RGBA32F:
			GLColorFormat = GL_RGBA32F; break;	
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
