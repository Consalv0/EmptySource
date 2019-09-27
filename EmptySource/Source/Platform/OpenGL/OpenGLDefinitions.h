#pragma once

namespace ESource {

	struct OpenGLInputTextureFormatInfo {
		unsigned int    InternalFormat,
			            InputFormat,
			            BlockType;
		// Is supported on the current platform
		bool            Supported;
		EPixelFormat    PixelFormat;
	};

	extern const OpenGLInputTextureFormatInfo OpenGLPixelFormatInfo[PF_MAX];

}