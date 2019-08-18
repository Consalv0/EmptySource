
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/RenderingAPI.h"
#include "Rendering/Shader.h"
#include "Rendering/Texture.h"

#include "Platform/OpenGL/OpenGLTexture.h"

#include "Rendering/Rendering.h"

namespace EmptySource {
	
	Texture2DPtr EmptySource::Texture2D::Create(
		const IntVector2 & Size, const EColorFormat ColorFormat, const EFilterMode & FilterMode,
		const ESamplerAddressMode & AddressMode) 
	{
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLTexture2D>(Size, ColorFormat, FilterMode, AddressMode);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create texture 2D!");
			return NULL;
		}
	}
	
	Texture2DPtr EmptySource::Texture2D::Create(
		const IntVector2 & Size, const EColorFormat ColorFormat, const EFilterMode & FilterMode,
		const ESamplerAddressMode & AddressMode, const EColorFormat InputFormat, const void * BufferData)
	{
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLTexture2D>(Size, ColorFormat, FilterMode, AddressMode, InputFormat, BufferData);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create texture 2D!");
			return NULL;
		}
	}

	bool Cubemap::ConvertFromCube(CubemapPtr & Cube, const TextureData<UCharRGB>& Textures) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return OpenGLCubemap::ConvertFromCube(Cube, Textures);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't convert cubemap from texture2D!");
			return NULL;
		}
	}

	bool Cubemap::ConvertFromHDRCube(CubemapPtr & Cube, const TextureData<FloatRGB>& Textures) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return OpenGLCubemap::ConvertFromHDRCube(Cube, Textures);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't convert cubemap from HDR texture2D!");
			return NULL;
		}
	}

	bool Cubemap::ConvertFromEquirectangular(CubemapPtr & Cube, Texture2DPtr Equirectangular, Material * EquirectangularToCubemapMaterial) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return OpenGLCubemap::ConvertFromEquirectangular(Cube, Equirectangular, EquirectangularToCubemapMaterial);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't convert cubemap from texture2D!");
			return NULL;
		}
	}

	bool Cubemap::ConvertFromHDREquirectangular(CubemapPtr & Cube, Texture2DPtr Equirectangular, Material * EquirectangularToCubemapMaterial) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return OpenGLCubemap::ConvertFromHDREquirectangular(Cube, Equirectangular, EquirectangularToCubemapMaterial);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't convert cubemap from texture2D!");
			return NULL;
		}
	}

	CubemapPtr EmptySource::Cubemap::Create(
		const unsigned int & Size, const EColorFormat & Format, const EFilterMode & Filter, const ESamplerAddressMode & AddressMode) 
	{
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLCubemap>(Size, Format, Filter, AddressMode);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create texture 2D!");
			return NULL;
		}
	}

}