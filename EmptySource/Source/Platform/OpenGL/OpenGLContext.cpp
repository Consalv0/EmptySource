#pragma once

#include "Graphics/GraphicContext.h"

struct SDL_Window;

namespace EmptySource {

	class OpenGLContext : public GraphicContext {
	public:
		OpenGLContext(SDL_Window * WindowHandle);

		virtual void Initialize() override;
		virtual void SwapBuffers() override;
	
	private:

		SDL_Window * mWindowHandle;
	
	};

}