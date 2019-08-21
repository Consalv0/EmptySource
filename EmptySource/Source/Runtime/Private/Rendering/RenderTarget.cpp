
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Shader.h"
#include "Rendering/Texture.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderingAPI.h"

#include "Rendering/Material.h"
#include "Mesh/Mesh.h"
#include "Mesh/MeshPrimitives.h"
#include "Math/Matrix4x4.h"
#include "Math/MathUtility.h"

#include "Platform/OpenGL/OpenGLTexture.h"
#include "Platform/OpenGL/OpenGLRenderTarget.h"
#include "Platform/OpenGL/OpenGLAPI.h"

#include "glad/glad.h"

namespace EmptySource {

	RenderTargetPtr RenderTarget::Create() {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLRenderTarget>();
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create render target!");
			return NULL;
		}
	}

}