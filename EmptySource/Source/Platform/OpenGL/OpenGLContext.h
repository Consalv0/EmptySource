#pragma once

#include "Rendering/GraphicContext.h"

struct SDL_Window;

namespace EmptySource {

	class OpenGLContext : public GraphicContext {
	private:

		//* OpenGL Context
		void * GLContext;

		//* OpenGL Version *.°
		unsigned int VersionMajor;
		//* OpenGL Version °.*
		unsigned int VersionMinor;
		
		//* Frame Count
		unsigned long long FrameCount;

		SDL_Window * WindowHandle;

	public:

		OpenGLContext(SDL_Window * Handle, unsigned int Major, unsigned int Minor);
		
		~OpenGLContext();

		virtual bool CreateContext();

		unsigned long long inline GetFrameCount() { return FrameCount; };

		virtual bool IsValid() override;

		virtual bool Initialize() override;

		virtual void SwapBuffers() override;
	
	};

}