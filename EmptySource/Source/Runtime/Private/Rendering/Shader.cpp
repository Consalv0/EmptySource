
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/RenderingAPI.h"
#include "Rendering/Shader.h"

#include "Platform/OpenGL/OpenGLShader.h"

#include "Rendering/Rendering.h"

namespace ESource {

	ShaderStage * ShaderStage::CreateFromText(const NString & Code, EShaderStageType Type) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return new OpenGLShaderStage(Code, Type);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create shader stage!");
			return NULL;
		}
	}

	ShaderProgram * ShaderProgram::Create(TArray<ShaderStage *> ShaderStages) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return new OpenGLShaderProgram(ShaderStages);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create shader!");
			return NULL;
		}
	}

}