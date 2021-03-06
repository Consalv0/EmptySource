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
		uint64_t FrameCount;
		
		NString DeviceName;

		SDL_Window * WindowHandle;

	public:

		OpenGLContext(SDL_Window * Handle, uint32_t Major, uint32_t Minor);
		
		~OpenGLContext();

		virtual bool CreateContext();

		virtual NString GetDeviceName() const override;

		virtual NString GetGLVersion() const override;

		virtual NString GetShaderVersion() const override;

		uint64_t inline GetFrameCount() const override { return FrameCount; };

		virtual bool IsValid() override;

		virtual bool Initialize() override;

		virtual void SwapBuffers() override;
	
	};

}