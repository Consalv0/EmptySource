
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"

namespace ESource {

	const PixelFormatInfo PixelFormats[PF_MAX] = {
		// Name               Size  Channels  Supported  EPixelFormat
		{ L"PF_Unknown",       0,      0,       false,    PF_Unknown       },
		{ L"PF_R8",            1,      1,        true,    PF_R8            },
		{ L"PF_R32F",          4,      1,        true,    PF_R32F          },
		{ L"PF_RG8",           2,      2,        true,    PF_RG8           },
		{ L"PF_RG32F",         8,      2,        true,    PF_RG32F         },
		{ L"PF_RG16F",         4,      2,        true,    PF_RG16F         },
		{ L"PF_RGB8",          3,      3,        true,    PF_RGB8          },
		{ L"PF_RGB32F",        12,     3,        true,    PF_RGB32F        },
		{ L"PF_RGBA8",         4,      4,        true,    PF_RGBA8         },
		{ L"PF_RGBA16_UShort", 8,      4,        true,    PF_RGBA16_UShort },
		{ L"PF_RGBA32F",       16,     4,        true,    PF_RGBA32F       },
		{ L"PF_DepthStencil",  4,      1,        true,    PF_DepthStencil  },
		{ L"PF_ShadowDepth",   4,      1,        true,    PF_ShadowDepth   }
	};

}