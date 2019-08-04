#pragma once

#include "Engine/Application.h"

#include "Graphics/GraphicContext.h"
#include "Platform/OpenGL/OpenGLContext.h"

#include "Utility/LogCore.h"
#include "Utility/LogGraphics.h"

// SDL 2.0.9
#include "../External/GLAD/include/glad/glad.h"

#include "../External/SDL2/include/SDL.h"
#include "../External/SDL2/include/SDL_opengl.h"

struct SDL_Window;

namespace EmptySource {

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
			Debug::Log(Debug::LogWarning, L"SDL_CreateContext failed: %ls\n", StringToWString(SDL_GetError()).c_str());
			return false;
		}

		return true;
	}

	bool OpenGLContext::IsValid() {
		return GLContext != NULL;
	}

	bool OpenGLContext::Initialize() {

		if (!gladLoadGLLoader(SDL_GL_GetProcAddress)) {
			Debug::Log(Debug::LogCritical, L"GL Functions could not be initialized: %ls",
				StringToWString(SDL_GetError()).c_str()
			);
			if (!gladLoadGL()) {
				Debug::Log(Debug::LogCritical, L"Unable to load OpenGL functions!");
				return false;
			}
		}

#ifndef __APPLE__
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