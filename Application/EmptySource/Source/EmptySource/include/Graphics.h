#pragma once

// Include GLAD
// Library to make function loaders for OpenGL
#include <External/GLAD/include/glad.h>

// Include GLFW
// Library to make crossplataform input and window creation
#include <External/GLFW/glfw3.h>

namespace Render {
	enum DepthFunction {
		Never = GL_NEVER,
		Less = GL_LESS,
		Equal = GL_EQUAL,
		LessEqual = GL_LEQUAL,
		Greater = GL_GREATER,
		NotEqual = GL_NOTEQUAL,
		GreaterEqual = GL_GEQUAL,
		Always = GL_ALWAYS
	};

	enum CullMode {
		None = GL_NONE,
		FrontLeft = GL_FRONT_LEFT,
		FrontRight = GL_FRONT_RIGHT,
		BackLeft = GL_BACK_LEFT,
		BackRight = GL_BACK_RIGHT,
		Front = GL_FRONT,
		Back = GL_BACK,
		Left = GL_LEFT,
		Right = GL_RIGHT,
		FrontBack = GL_FRONT_AND_BACK,
	};

	enum RenderMode {
		Point = GL_POINT,
		Wire = GL_LINE,
		Fill = GL_FILL,
	};
}