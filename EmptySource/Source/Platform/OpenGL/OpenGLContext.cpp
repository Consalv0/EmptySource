#pragma once

#include "CoreMinimal.h"
#include "Platform/OpenGL/OpenGLContext.h"

#include "Utility/TextFormatting.h"

#include <glad/glad.h>

// SDL 2.0.9
#include <SDL.h>
#include <SDL_opengl.h>


namespace ESource {

	namespace Debug {

#ifndef ES_PLATFORM_APPLE
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

			LOG_CORE_ERROR(L"<{0}>({1:d}) {2}", ErrorPrefix, ErrorID, Text::NarrowToWide(ErrorMessage));
		}
#endif

		//* Prints the GPU version info used by GL
		void PrintGraphicsInformation() {
			const GLubyte    *Renderer = glGetString(GL_RENDERER);
			const GLubyte      *Vendor = glGetString(GL_VENDOR);
			const GLubyte     *Version = glGetString(GL_VERSION);
			const GLubyte *GLSLVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

			LOG_CORE_INFO("GPU Render Device Info");
			LOG_CORE_INFO(" - GC Vendor	: {0}", (const NChar*)Vendor);
			LOG_CORE_INFO(" - GC Renderer	: {0}", (const NChar*)Renderer);
			LOG_CORE_INFO(" - GL Version	: {0}", (const NChar*)Version);
			LOG_CORE_INFO(" - GLSL Version	: {0}", (const NChar*)GLSLVersion);
		}
	}

	OpenGLContext::~OpenGLContext() {
		SDL_GL_DeleteContext(GLContext);
	}

	OpenGLContext::OpenGLContext(SDL_Window * Handle, unsigned int Major, unsigned int Minor) {
		GLContext = NULL;
		WindowHandle = Handle;
		VersionMajor = Major;
		VersionMinor = Minor;

		CreateContext();
	}

	bool OpenGLContext::CreateContext() {
		if (GLContext != NULL) return false;

		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, VersionMajor);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, VersionMinor);
		SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

		GLContext = SDL_GL_CreateContext(WindowHandle);

		if (GLContext == NULL) {
			LOG_CORE_CRITICAL("SDL_CreateContext failed: {0}", SDL_GetError());
			return false;
		}

		return true;
	}

	bool OpenGLContext::IsValid() {
		return GLContext != NULL;
	}

	bool OpenGLContext::Initialize() {

		if (!gladLoadGLLoader(SDL_GL_GetProcAddress)) {
			LOG_CORE_CRITICAL("GL Functions could not be initialized: {0}", SDL_GetError());
			if (!gladLoadGL()) {
				LOG_CORE_CRITICAL("Unable to load OpenGL functions!");
				return false;
			}
		}

#ifndef ES_PLATFORM_APPLE
		glEnable(GL_DEBUG_OUTPUT);
		// glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(Debug::OGLError, nullptr);
		// --- Enable all messages, all sources, all levels, and all IDs:
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
#endif

		Debug::PrintGraphicsInformation();

		return true;
	}

	void OpenGLContext::SwapBuffers() {
		SDL_GL_SwapWindow(WindowHandle);
		FrameCount++;
	}

}