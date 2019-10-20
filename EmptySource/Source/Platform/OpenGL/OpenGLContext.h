#pragma once

#include "Rendering/GraphicContext.h"

struct SDL_Window;

namespace ESource {

	class OpenGLContext : public GraphicContext {
	private:

		//* OpenGL Context
		void * GLContext;

		//* OpenGL Version *.°
		uint32_t VersionMajor;
		//* OpenGL Version °.*
		uint32_t VersionMinor;
		
		//* Frame Count
		unsigned long long FrameCount;

		SDL_Window * WindowHandle;

	public:

		OpenGLContext(SDL_Window * Handle, uint32_t Major, uint32_t Minor);
		
		~OpenGLContext();

		virtual bool CreateContext();

		unsigned long long inline GetFrameCount() { return FrameCount; };

		virtual bool IsValid() override;

		virtual bool Initialize() override;

		virtual void SwapBuffers() override;
	
	};

}