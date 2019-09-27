
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Platform/OpenGL/OpenGLDefinitions.h"

#include "glad/glad.h"

namespace ESource {

	const PixelFormatInfo PixelFormats[PF_MAX] = {
		// Name               Size  Channels  PlatformFormat         Supported   EPixelFormat
		{ L"PF_Unknown",       0,      0,      GL_NONE,                false,     PF_Unknown       },
		{ L"PF_R8",            1,      1,      GL_R8,                   true,     PF_R8            },
		{ L"PF_R32F",          4,      1,      GL_R32F,                 true,     PF_R32F          },
		{ L"PF_RG8",           2,      2,      GL_RG8,                  true,     PF_RG8           },
		{ L"PF_RG32F",         8,      2,      GL_RG32F,                true,     PF_RG32F         },
		{ L"PF_RG16F",         4,      2,      GL_RG16F,                true,     PF_RG16F         },
		{ L"PF_RGB8",          3,      3,      GL_RGB8,                 true,     PF_RGB8          },
		{ L"PF_RGB32F",        12,     3,      GL_RGB32F,               true,     PF_RGB32F        },
		{ L"PF_RGBA8",         4,      4,      GL_RGBA8,                true,     PF_RGBA8         },
		{ L"PF_RGBA16_UShort", 8,      4,      GL_RGBA16,               true,     PF_RGBA16_UShort },
		{ L"PF_RGBA32F",       16,     4,      GL_RGBA32F,              true,     PF_RGBA32F       },
		{ L"PF_DepthStencil",  4,      1,      GL_DEPTH_COMPONENT32F,   true,     PF_DepthStencil  },
		{ L"PF_ShadowDepth",   4,      1,      GL_DEPTH32F_STENCIL8,    true,     PF_ShadowDepth   }
	};

	const OpenGLInputTextureFormatInfo OpenGLPixelFormatInfo[PF_MAX] = {
		//PlatformFormat         InputFormat   BlockType            EPixelFormat
		{  GL_NONE,               GL_NONE,      GL_NONE,             PF_Unknown       },
		{  GL_R8,                 GL_RED,       GL_UNSIGNED_BYTE,    PF_R8            },
		{  GL_R32F,               GL_RED,       GL_FLOAT,            PF_R32F          },
		{  GL_RG8,                GL_RG,        GL_UNSIGNED_BYTE,    PF_RG8           },
		{  GL_RG32F,              GL_RG,        GL_FLOAT,            PF_RG32F         },
		{  GL_RG16F,              GL_RG,        GL_FLOAT,            PF_RG16F         },
		{  GL_RGB8,               GL_RGB,       GL_UNSIGNED_BYTE,    PF_RGB8          },
		{  GL_RGB32F,             GL_RGB,       GL_FLOAT,            PF_RGB32F        },
		{  GL_RGBA8,              GL_RGBA,      GL_UNSIGNED_BYTE,    PF_RGBA8         },
		{  GL_RGBA16,             GL_RGBA,      GL_UNSIGNED_SHORT,   PF_RGBA16_UShort },
		{  GL_RGBA32F,            GL_RGBA,      GL_FLOAT,            PF_RGBA32F       },
		{  GL_DEPTH_COMPONENT32F, GL_RED,       GL_FLOAT,            PF_DepthStencil  },
		{  GL_DEPTH32F_STENCIL8,  GL_RED,       GL_FLOAT,            PF_ShadowDepth   }

	};

}