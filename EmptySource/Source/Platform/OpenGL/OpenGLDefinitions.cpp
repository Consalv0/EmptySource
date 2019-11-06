
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Platform/OpenGL/OpenGLDefinitions.h"

#include "glad/glad.h"

namespace ESource {

	const OpenGLInputTextureFormatInfo OpenGLPixelFormatInfo[PF_MAX] = {
		//PlatformFormat         InputFormat          BlockType           Supported  EPixelFormat
		{  GL_NONE,               GL_NONE,             GL_NONE,             false,    PF_Unknown          },
		{  GL_R8,                 GL_RED,              GL_UNSIGNED_BYTE,     true,    PF_R8               },
		{  GL_R32F,               GL_RED,              GL_FLOAT,             true,    PF_R32F             },
		{  GL_RG8,                GL_RG,               GL_UNSIGNED_BYTE,     true,    PF_RG8              },
		{  GL_RG32F,              GL_RG,               GL_FLOAT,             true,    PF_RG32F            },
		{  GL_RG16F,              GL_RG,               GL_FLOAT,             true,    PF_RG16F            },
		{  GL_RGB8,               GL_RGB,              GL_UNSIGNED_BYTE,     true,    PF_RGB8             },
		{  GL_RGB32F,             GL_RGB,              GL_FLOAT,             true,    PF_RGB32F           },
		{  GL_RGB16F,             GL_RGB,              GL_FLOAT,             true,    PF_RGB16F           },
		{  GL_RGBA8,              GL_RGBA,             GL_UNSIGNED_BYTE,     true,    PF_RGBA8            },
		{  GL_RGBA16,             GL_RGBA,             GL_UNSIGNED_SHORT,    true,    PF_RGBA16_UShort    },
		{  GL_RGBA32F,            GL_RGBA,             GL_FLOAT,             true,    PF_RGBA32F          },
		{  GL_DEPTH_COMPONENT24,  GL_DEPTH_COMPONENT,  GL_FLOAT,             true,    PF_DepthComponent24 },
		{  GL_DEPTH24_STENCIL8,   GL_DEPTH_STENCIL,    GL_UNSIGNED_INT_24_8, true,    PF_DepthStencil     },
		{  GL_DEPTH_COMPONENT,    GL_DEPTH_COMPONENT,  GL_FLOAT,             true,    PF_ShadowDepth      }

	}; 

}