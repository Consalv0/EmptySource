#pragma once

// --- Include GLAD library to make function loaders for OpenGL
#ifndef __gl_h_
#include "../GLAD/include/glad/glad.h"
#endif // !__gl_h_

namespace Debug {
	//* Error Callback related to GLFW
	inline void GLFWError(int ErrorID, const char* Description) {
		Debug::Log(Debug::LogError, L"%ls", CharToWString(Description).c_str());
	}
    
#ifndef __APPLE__
	//* Error Callback related to OpenGL > 4.3
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

		Debug::Log(Debug::LogError, L"<%ls>(%i) %ls", ErrorPrefix, ErrorID, CharToWString(ErrorMessage).c_str());
	}
#endif
    
	//* Prints the GPU version info used by GL
	inline void PrintGraphicsInformation() {
		const GLubyte    *Renderer = glGetString(GL_RENDERER);
		const GLubyte      *Vendor = glGetString(GL_VENDOR);
		const GLubyte     *Version = glGetString(GL_VERSION);
		const GLubyte *GLSLVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

		Log(LogInfo, L"GPU Render Device Info");
		Log(LogInfo, L"├> GC Vendor	: %ls", CharToWString((const char*)Vendor).c_str());
		Log(LogInfo, L"├> GC Renderer	: %ls", CharToWString((const char*)Renderer).c_str());
		Log(LogInfo, L"├> GL Version	: %ls", CharToWString((const char*)Version).c_str());
		Log(LogInfo, L"└> GLSL Version	: %ls", CharToWString((const char*)GLSLVersion).c_str());
	}
}
