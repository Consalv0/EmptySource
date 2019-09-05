
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
		const WString& Name, const IntVector2 & Size, const EColorFormat ColorFormat, const EFilterMode & FilterMode,
		const ESamplerAddressMode & AddressMode) 
	{
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLTexture2D>(Name, Size, ColorFormat, FilterMode, AddressMode);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create texture 2D!");
			return NULL;
		}
	}
	
	Texture2DPtr EmptySource::Texture2D::Create(
		const WString& Name, const IntVector2 & Size, const EColorFormat ColorFormat, const EFilterMode & FilterMode,
		const ESamplerAddressMode & AddressMode, const EColorFormat InputFormat, const void * BufferData)
	{
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLTexture2D>(Name, Size, ColorFormat, FilterMode, AddressMode, InputFormat, BufferData);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create texture 2D!");
			return NULL;
		}
	}

	CubemapPtr EmptySource::Cubemap::Create(
		const WString& Name, const unsigned int & Size, const EColorFormat & Format, const EFilterMode & Filter, const ESamplerAddressMode & AddressMode)
	{
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLCubemap>(Name, Size, Format, Filter, AddressMode);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create texture 2D!");
			return NULL;
		}
	}

}