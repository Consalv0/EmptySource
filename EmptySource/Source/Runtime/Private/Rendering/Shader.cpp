
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/RenderingAPI.h"
#include "Rendering/Shader.h"

#include "Platform/OpenGL/OpenGLShader.h"

#include "Rendering/Rendering.h"

namespace EmptySource {

	ShaderStagePtr ShaderStage::CreateFromFile(const WString & FilePath, EShaderType Type) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLShaderStage>(FilePath, Type);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create shader stage!");
			return NULL;
		}
	}

	ShaderStagePtr ShaderStage::CreateFromText(const NString & Code, EShaderType Type) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLShaderStage>(Code, Type);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create shader stage!");
			return NULL;
		}
	}

	ShaderPtr ShaderProgram::Create(const WString& Name, TArray<ShaderStagePtr> ShaderStages) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return std::make_shared<OpenGLShaderProgram>(Name, ShaderStages);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create shader!");
			return NULL;
		}
	}

}