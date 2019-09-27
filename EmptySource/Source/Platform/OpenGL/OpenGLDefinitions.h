#pragma once

namespace EmptySource {

	struct OpenGLInputTextureFormatInfo {
		unsigned int    InternalFormat,
			            InputFormat,
			            BlockType;
		EPixelFormat    PixelFormat;
	};

	extern const OpenGLInputTextureFormatInfo OpenGLPixelFormatInfo[PF_MAX];

}