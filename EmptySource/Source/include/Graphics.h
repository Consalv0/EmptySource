#pragma once

typedef struct UCharRed  { typedef unsigned char Range; static constexpr unsigned char Channels = 1; Range R; } UCharRed;
typedef struct UCharRG   { typedef unsigned char Range; static constexpr unsigned char Channels = 2; Range R; Range G; } UCharRG;
typedef struct UCharRGB  { typedef unsigned char Range; static constexpr unsigned char Channels = 3; Range R; Range G; Range B; } UCharRGB;
typedef struct UCharRGBA { typedef unsigned char Range; static constexpr unsigned char Channels = 4; Range R; Range G; Range B; Range A; } UCharRGBA;
typedef struct FloatRed  { typedef float Range; static constexpr unsigned char Channels = 1; Range R; } FloatRed;
typedef struct FloatRG   { typedef float Range; static constexpr unsigned char Channels = 2; Range R; Range G; } FloatRG;
typedef struct FloatRGB  { typedef float Range; static constexpr unsigned char Channels = 3; Range R; Range G; Range B; } FloatRGB;
typedef struct FloatRGBA { typedef float Range; static constexpr unsigned char Channels = 4; Range R; Range G; Range B; Range A; } FloatRGBA;

namespace Graphics {
	enum DepthFunction {
		DF_Never,
		DF_Less,
		DF_Equal,
		DF_LessEqual,
		DF_Greater,
		DF_NotEqual,
		DF_GreaterEqual,
		DF_Always
	};

	enum CullMode {
		CM_None,
	 // CM_FrontLeft,
 	 // CM_FrontRight,
     // CM_BackLeft,
	 // CM_BackRight,
		CM_Front,
		CM_Back,
	 // CM_Left,
	 // CM_Right,
		CM_FrontBack,
	};

	enum RenderMode {
		RM_Point,
		RM_Wire,
		RM_Fill,
	};

	enum ColorFormat {
		CF_Red,
		CF_RG,
		CF_RGB,
		CF_RGBA,
		CF_RG16F,
		CF_RGBA16F,
		CF_RGB16F,
		CF_RGBA32F,
		CF_RGB32F,
	};

	enum AddressMode {
		AM_Repeat,
		AM_Mirror,
		AM_Clamp,
		AM_Border,
	};

	enum FilterMode {
		FM_MinMagNearest,
		FM_MinMagLinear,
		FM_MinLinearMagNearest,
		FM_MinNearestMagLinear,
	};

	enum DrawMode {
		DM_Points,
		DM_Lines,
		DM_Triangles
	};
}
