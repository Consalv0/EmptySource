
#include "../include/Core.h"
#include "../include/Texture.h"
#include "../include/GLFunctions.h"

unsigned int Texture::GetColorFormat(const Graphics::ColorFormat & CF) {
	switch (CF) {
		case Graphics::CF_Red:
			return GL_RED;
		case Graphics::CF_RG:
			return GL_RG;
		case Graphics::CF_RGB:
			return GL_RGB;
		case Graphics::CF_RGBA:
			return GL_RGBA;
		case Graphics::CF_RG16F:
			return GL_RG16F;
		case Graphics::CF_RGB16F:
			return GL_RGB16F;
		case Graphics::CF_RGBA16F:
			return GL_RGBA16F;
		case Graphics::CF_RGBA32F:
			return GL_RGBA32F;
		case Graphics::CF_RGB32F:
			return GL_RGB32F;
		default:
			Debug::Log(Debug::LogWarning, L"Color not implemented, using RGBA");
			return GL_RGBA;
	}
}

unsigned int Texture::GetColorFormatInput(const Graphics::ColorFormat & CF) {
	switch (CF) {
		case Graphics::CF_Red:
			return GL_RED;
		case Graphics::CF_RG16F:
		case Graphics::CF_RG:
			return GL_RG;
		case Graphics::CF_RGB32F:
		case Graphics::CF_RGB16F:
		case Graphics::CF_RGB:
			return GL_RGB;
		case Graphics::CF_RGBA32F:
		case Graphics::CF_RGBA16F:
		case Graphics::CF_RGBA:
			return GL_RGBA;
		default:
			Debug::Log(Debug::LogWarning, L"Color not implemented, using RGBA");
			return GL_RGBA;
	}
}

unsigned int Texture::GetInputType(const Graphics::ColorFormat & CF) {
	switch (CF) {
		case Graphics::CF_Red:
		case Graphics::CF_RG:
		case Graphics::CF_RGB:
		case Graphics::CF_RGBA:
			return GL_UNSIGNED_BYTE;
		case Graphics::CF_RG16F:
		case Graphics::CF_RGB16F:
		case Graphics::CF_RGBA16F:
		case Graphics::CF_RGBA32F:
		case Graphics::CF_RGB32F:
			return GL_FLOAT;
		default:
			Debug::Log(Debug::LogWarning, L"Color not implemented, using unsigned byte");
			return GL_UNSIGNED_BYTE;
	}
}

Texture::Texture() {
	TextureObject = 0;
	FilterMode = Graphics::FM_MinMagLinear;
	AddressMode = Graphics::AM_Border;
	ColorFormat = Graphics::CF_RGB;
}
