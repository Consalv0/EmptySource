
#include "CoreMinimal.h"
#include "Math/CoreMath.h"
#include "Rendering/Rendering.h"
#include "Platform/OpenGL/OpenGLAPI.h"


namespace ESource {

	RenderingAPI * Rendering::RendererAppInterface = new OpenGLAPI();

}