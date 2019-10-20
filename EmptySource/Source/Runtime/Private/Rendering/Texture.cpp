
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/RenderingAPI.h"
#include "Rendering/Shader.h"
#include "Rendering/Texture.h"

#include "Platform/OpenGL/OpenGLTexture.h"

#include "Rendering/Rendering.h"

namespace ESource {
	
	Texture2D * ESource::Texture2D::Create(
		const IntVector2 & Size, const EPixelFormat ColorFormat, const EFilterMode & FilterMode,
		const ESamplerAddressMode & AddressMode) 
	{
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return new OpenGLTexture2D(Size, ColorFormat, FilterMode, AddressMode);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create texture 2D!");
			return NULL;
		}
	}
	
	Texture2D * ESource::Texture2D::Create(
		const IntVector2 & Size, const EPixelFormat ColorFormat, const EFilterMode & FilterMode,
		const ESamplerAddressMode & AddressMode, const EPixelFormat InputFormat, const void * BufferData)
	{
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return new OpenGLTexture2D(Size, ColorFormat, FilterMode, AddressMode, InputFormat, BufferData);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create texture 2D!");
			return NULL;
		}
	}

	Cubemap * ESource::Cubemap::Create(
		const uint32_t & Size, const EPixelFormat & Format, const EFilterMode & Filter, const ESamplerAddressMode & AddressMode)
	{
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return new OpenGLCubemap(Size, Format, Filter, AddressMode);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create texture 2D!");
			return NULL;
		}
	}

}