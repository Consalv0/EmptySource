#pragma once

// --- Include GLAD library to make function loaders for OpenGL
#include <External/GLAD/include/glad.h>

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

#ifdef LOG_CORE

namespace Debug {
	//* Error Callback related to GLFW
	inline void GLFWError(int ErrorID, const char* Description) {
		Debug::Log(Debug::LogError, L"%s", CharToWChar(Description));
	}

	//* Error Callback related to OpenGL
	inline void APIENTRY OGLError(
		GLenum ErrorSource, GLenum ErrorType, GLuint ErrorID, GLenum ErrorSeverity, GLsizei ErrorLength,
		const GLchar * ErrorMessage, const void * UserParam)
	{
		// --- Ignore non-significant error/warning codes
		if (ErrorID == 131169 || ErrorID == 131185 || ErrorID == 131218 || ErrorID == 131204) return;

		const WChar* ErrorPrefix = L"";

		switch (ErrorType) {
		case GL_DEBUG_TYPE_ERROR:               ErrorPrefix = L"error";       break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: ErrorPrefix = L"deprecated";  break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  ErrorPrefix = L"undefined";   break;
		case GL_DEBUG_TYPE_PORTABILITY:         ErrorPrefix = L"portability"; break;
		case GL_DEBUG_TYPE_PERFORMANCE:         ErrorPrefix = L"performance"; break;
		case GL_DEBUG_TYPE_MARKER:              ErrorPrefix = L"marker";      break;
		case GL_DEBUG_TYPE_PUSH_GROUP:          ErrorPrefix = L"pushgroup";  break;
		case GL_DEBUG_TYPE_POP_GROUP:           ErrorPrefix = L"popgroup";   break;
		case GL_DEBUG_TYPE_OTHER:               ErrorPrefix = L"other";       break;
		}

		Debug::Log(Debug::LogError, L"<%s>(%i) %s", ErrorPrefix, ErrorID, CharToWChar(ErrorMessage));
	}

	//* Prints the GPU version info used by GL
	inline void PrintGraphicsInformation() {
		const GLubyte    *renderer = glGetString(GL_RENDERER);
		const GLubyte      *vendor = glGetString(GL_VENDOR);
		const GLubyte     *version = glGetString(GL_VERSION);
		const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

		Log(LogNormal, L"GPU Render Device Info");
		Log(LogNormal, L"├> GC Vendor	: %s", CharToWChar((const char*)vendor));
		Log(LogNormal, L"├> GC Renderer	: %s", CharToWChar((const char*)renderer));
		Log(LogNormal, L"├> GL Version	: %s", CharToWChar((const char*)version));
		Log(LogNormal, L"└> GLSL Version	: %s", CharToWChar((const char*)glslVersion));
	}
}
#endif // LOG_CORE