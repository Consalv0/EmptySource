#pragma once

namespace EmptySource {

	typedef struct UCharRed  { typedef unsigned char Range; static constexpr unsigned char Channels = 1; Range R; } UCharRed;
	typedef struct UCharRG   { typedef unsigned char Range; static constexpr unsigned char Channels = 2; Range R; Range G; } UCharRG;
	typedef struct UCharRGB  { typedef unsigned char Range; static constexpr unsigned char Channels = 3; Range R; Range G; Range B; } UCharRGB;
	typedef struct UCharRGBA { typedef unsigned char Range; static constexpr unsigned char Channels = 4; Range R; Range G; Range B; Range A; } UCharRGBA;
	typedef struct FloatRed  { typedef float Range; static constexpr unsigned char Channels = 1; Range R; } FloatRed;
	typedef struct FloatRG   { typedef float Range; static constexpr unsigned char Channels = 2; Range R; Range G; } FloatRG;
	typedef struct FloatRGB  { typedef float Range; static constexpr unsigned char Channels = 3; Range R; Range G; Range B; } FloatRGB;
	typedef struct FloatRGBA { typedef float Range; static constexpr unsigned char Channels = 4; Range R; Range G; Range B; Range A; } FloatRGBA;

	enum class ETextureDimension {
		None,
		Texture1D,
		Texture2D, 
		Texture3D, 
		Cubemap
	};

	enum class ECubemapFace {
		Front, 
		Back, 
		Right,
		Left,
		Up,
		Down
	};

	enum class EShaderDataType {
		None = 0,
		Bool,
		Float,
		Float2,
		Float3,
		Float4,
		Int,
		Int2,
		Int3, 
		Int4, 
		Mat3x3,
		Mat4x4
	};

	enum EShaderType {
		ST_Vertex,
		ST_Geometry,
		ST_Pixel,
		ST_Compute
	};

	enum EBlendFactor {
		BF_Zero,
		BF_One,
		BF_SourceColor,
		BF_InverseSourceColor,
		BF_SourceAlpha,
		BF_InverseSourceAlpha,
		BF_DestAlpha,
		BF_InverseDestAlpha,
		BF_DestColor,
		BF_InverseDestColor,
		BF_ConstantBlendFactor,
		BF_InverseConstantBlendFactor
	};

	enum EUsageMode {
		UM_Default,
		UM_Static,
		UM_Dynamic,
		UM_Inmutable,
	};

	enum EDepthFunction {
		DF_Never,
		DF_Less,
		DF_Equal,
		DF_LessEqual,
		DF_Greater,
		DF_NotEqual,
		DF_GreaterEqual,
		DF_Always
	};

	enum ECullMode {
		CM_None,
		CM_ClockWise,
		CM_CounterClockWise
	};

	enum EColorFormat {
		CF_Red,
		CF_RG,
		CF_RG16F,
		CF_RGB,
		CF_RGB16F,
		CF_RGB32F,
		CF_RGBA,
		CF_RGBA16F,
		CF_RGBA32F
	};

	enum ESamplerAddressMode {
		SAM_Repeat,
		SAM_Mirror,
		SAM_Clamp,
		SAM_Border
	};

	enum ERasterizerFillMode {
		FM_Point,
		FM_Wireframe,
		FM_Solid,
	};

	enum EFilterMode {
		FM_MinMagNearest,
		FM_MinMagLinear,
		FM_MinLinearMagNearest,
		FM_MinNearestMagLinear,
	};

	enum EDrawMode {
		DM_Points,
		DM_Lines,
		DM_Triangles
	};

}