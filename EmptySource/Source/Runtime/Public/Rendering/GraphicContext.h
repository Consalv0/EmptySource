#pragma once

namespace EmptySource {

	class GraphicContext {
	public:

		virtual bool IsValid() = 0;

		virtual bool Initialize() = 0;

		virtual void SwapBuffers() = 0;
	
	};

}