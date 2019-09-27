#pragma once

namespace ESource {

	struct OpenGLInputTextureFormatInfo {
		unsigned int    InternalFormat,
			            InputFormat,
			            BlockType;
		EPixelFormat    PixelFormat;
	};

	extern const OpenGLInputTextureFormatInfo OpenGLPixelFormatInfo[PF_MAX];

}