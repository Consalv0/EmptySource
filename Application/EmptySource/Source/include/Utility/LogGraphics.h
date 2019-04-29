#pragma once

// --- Include GLAD library to make function loaders for OpenGL
#ifndef __gl_h_
#include "External/GLAD/glad.h"
#endif // !__gl_h_

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

		Log(LogInfo, L"GPU Render Device Info");
		Log(LogInfo, L"├> GC Vendor	: %ls", CharToWChar((const char*)Vendor));
		Log(LogInfo, L"├> GC Renderer	: %ls", CharToWChar((const char*)Renderer));
		Log(LogInfo, L"├> GL Version	: %ls", CharToWChar((const char*)Version));
		Log(LogInfo, L"└> GLSL Version	: %ls", CharToWChar((const char*)GLSLVersion));
	}

	inline void DebugBox(float MinX, float MinY, float MinZ, float MaxX, float MaxY, float MaxZ, Matrix4x4 * Transform) {
		GLuint ModelMatrixBuffer;
		glGenBuffers(1, &ModelMatrixBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);

		Vector3 size = Vector3(MaxX - MinX, MaxY - MinY, MaxZ - MinZ);
		Vector3 center = Vector3((MinX + MaxX) / 2, (MinY + MaxY) / 2, (MinZ + MaxZ) / 2);
		Matrix4x4 transform = Matrix4x4::Translation(center) * Matrix4x4::Scaling(size);
	}
}
