
#include "CoreMinimal.h"
#include "Rendering/Texture.h"
#include "Rendering/GLFunctions.h"

namespace EmptySource {

	unsigned int Texture::GetColorFormat(const EColorFormat & CF) {
		switch (CF) {
		case CF_Red:
			return GL_RED;
		case CF_RG:
			return GL_RG;
		case CF_RGB:
			return GL_RGB;
		case CF_RGBA:
			return GL_RGBA;
		case CF_RG16F:
			return GL_RG16F;
		case CF_RGB16F:
			return GL_RGB16F;
		case CF_RGBA16F:
			return GL_RGBA16F;
		case CF_RGBA32F:
			return GL_RGBA32F;
		case CF_RGB32F:
			return GL_RGB32F;
		default:
			LOG_CORE_WARN(L"Color not implemented, using RGBA");
			return GL_RGBA;
		}
	}

	unsigned int Texture::GetColorFormatInput(const EColorFormat & CF) {
		switch (CF) {
		case CF_Red:
			return GL_RED;
		case CF_RG16F:
		case CF_RG:
			return GL_RG;
		case CF_RGB32F:
		case CF_RGB16F:
		case CF_RGB:
			return GL_RGB;
		case CF_RGBA32F:
		case CF_RGBA16F:
		case CF_RGBA:
			return GL_RGBA;
		default:
			LOG_CORE_WARN(L"Color not implemented, using RGBA");
			return GL_RGBA;
		}
	}

	unsigned int Texture::GetInputType(const EColorFormat & CF) {
		switch (CF) {
		case CF_Red:
		case CF_RG:
		case CF_RGB:
		case CF_RGBA:
			return GL_UNSIGNED_BYTE;
		case CF_RG16F:
		case CF_RGB16F:
		case CF_RGBA16F:
		case CF_RGBA32F:
		case CF_RGB32F:
			return GL_FLOAT;
		default:
			LOG_CORE_WARN(L"Color not implemented, using unsigned byte");
			return GL_UNSIGNED_BYTE;
		}
	}

	Texture::Texture() {
		TextureObject = 0;
		FilterMode = FM_MinMagLinear;
		AddressMode = SAM_Border;
		ColorFormat = CF_RGB;
	}

}