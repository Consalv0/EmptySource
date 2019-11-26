#pragma once

namespace ESource {

	class GraphicContext {
	public:
		virtual bool IsValid() = 0;

		virtual bool Initialize() = 0;

		virtual void SwapBuffers() = 0;

		virtual NString GetDeviceName() const = 0;

		virtual NString GetGLVersion() const = 0;

		virtual NString GetShaderVersion() const = 0;
	
		virtual uint64_t inline GetFrameCount() const = 0;

	};

}