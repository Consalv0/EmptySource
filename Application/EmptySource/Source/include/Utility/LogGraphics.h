#pragma once

// --- Include GLAD library to make function loaders for OpenGL
#ifndef __gl_h_
#include "External/GLAD/glad.h"
#endif // !__gl_h_

#ifdef LOG_CORE
namespace Debug {
	//* Error Callback related to GLFW
	inline void GLFWError(int ErrorID, const char* Description) {
		Debug::Log(Debug::LogError, L"%ls", CharToWChar(Description));
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

		Debug::Log(Debug::LogError, L"<%ls>(%i) %ls", ErrorPrefix, ErrorID, CharToWChar(ErrorMessage));
	}
#endif
    
	//* Prints the GPU version info used by GL
	inline void PrintGraphicsInformation() {
		const GLubyte    *Renderer = glGetString(GL_RENDERER);
		const GLubyte      *Vendor = glGetString(GL_VENDOR);
		const GLubyte     *Version = glGetString(GL_VERSION);
		const GLubyte *GLSLVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

		Log(LogNormal, L"GPU Render Device Info");
		Log(LogNormal, L"├> GC Vendor	: %ls", CharToWChar((const char*)Vendor));
		Log(LogNormal, L"├> GC Renderer	: %ls", CharToWChar((const char*)Renderer));
		Log(LogNormal, L"├> GL Version	: %ls", CharToWChar((const char*)Version));
		Log(LogNormal, L"└> GLSL Version	: %ls", CharToWChar((const char*)GLSLVersion));
	}
}
#endif // LOG_CORE
