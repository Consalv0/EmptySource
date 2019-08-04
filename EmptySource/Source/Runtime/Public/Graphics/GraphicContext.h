#pragma once

namespace EmptySource {

	class GraphicContext {
	public:

		virtual void Initialize() = 0;

		virtual void SwapBuffers() = 0;
	
	};

}