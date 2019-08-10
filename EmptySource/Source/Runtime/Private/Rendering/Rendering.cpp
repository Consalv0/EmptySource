
#include "Engine/Log.h"
#include "Engine/Core.h"
#include "Math/CoreMath.h"
#include "Rendering/Rendering.h"
#include "Platform/OpenGL/OpenGLAPI.h"


namespace EmptySource {

	RenderingAPI * Rendering::RendererAppInterface = new OpenGLAPI();

}