#pragma once

typedef unsigned char UCharRed;
typedef struct { unsigned char R; unsigned char G; } UCharRG;
typedef struct { unsigned char R; unsigned char G; unsigned char B; } UCharRGB;
typedef struct { unsigned char R; unsigned char G; unsigned char B; unsigned char A; } UCharRGBA;
typedef float FloatRed;
typedef struct { float R; float G; } FloatRG;
typedef struct { float R; float G; float B; } FloatRGB;
typedef struct { float R; float G; float B; float A; } FloatRGBA;

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