#pragma once

#include "../include/Math/CoreMath.h"

namespace EmptySource {
	class ApplicationRenderer {
	public:
		enum class GLAPI {
			GLAPI_None = 0, GLAPI_OpenGL = 1, GLAPI_Vulkan = 2
		};

		virtual void SetClearColor(const Vector4& Color) = 0;
		virtual void Clear() const = 0;

		virtual void DrawIndexed() = 0;

		inline static GLAPI GetAPI() { 
			static GLAPI GraphicsLibraryAPI;
			return GraphicsLibraryAPI;
		};

	private:
	};
}