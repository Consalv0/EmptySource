#pragma once

namespace ESource {
	
	typedef struct UCharRed  { typedef unsigned char Range; static constexpr unsigned char Channels = 1; Range R; } UCharRed;
	typedef struct UCharRG   { typedef unsigned char Range; static constexpr unsigned char Channels = 2; Range R; Range G; } UCharRG;
	typedef struct UCharRGB  { typedef unsigned char Range; static constexpr unsigned char Channels = 3; Range R; Range G; Range B; } UCharRGB;
	typedef struct UCharRGBA { typedef unsigned char Range; static constexpr unsigned char Channels = 4; Range R; Range G; Range B; Range A; } UCharRGBA;
	typedef struct UShortRG  { typedef unsigned char Range; static constexpr unsigned char Channels = 2; Range R; Range G; } UShortRG;
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

	enum class EShaderUniformType {
		None = 0,
		Matrix4x4Array,
		Matrix4x4,
		FloatArray,
		Float,
		Float2DArray,
		Float2D,
		Float3DArray,
		Float3D,
		Float4DArray,
		Float4D,
		Texture2D,
		Cubemap,
		Int,
		IntArray,
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

	enum EShaderStageType {
		ST_Unknown,
		ST_Vertex,
		ST_Geometry,
		ST_Pixel,
		ST_Compute
	};

	enum EBlendFactor {
		BF_None,
		BF_Zero,
		BF_One,
		BF_SrcColor,
		BF_SrcAlpha,
		BF_DstAlpha,
		BF_DstColor,
		BF_OneMinusSrcColor,
		BF_OneMinusSrcAlpha,
		BF_OneMinusDstAlpha,
		BF_OneMinusDstColor,
	};

	enum EUsageMode {
		UM_Default,
		UM_Static,
		UM_Dynamic,
		UM_Inmutable,
		UM_Draw
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

	enum EStencilFunction {
		SF_Never,
		SF_Less,
		SF_Equal,
		SF_LessEqual,
		SF_Greater,
		SF_NotEqual,
		SF_GreaterEqual,
		SF_Always
	};

	enum EStencilOperation {
		SO_Keep,
		SO_Zero,
		SO_Replace,
		SO_Increment,
		SO_IncrementLoop,
		SO_Decrement,
		SO_DecrementLoop,
		SO_Invert
	};

	enum ECullMode {
		CM_None,
		CM_ClockWise,
		CM_CounterClockWise
	};

	enum EPixelFormat {
		PF_Unknown,
		PF_R8,
		PF_R32F,
		PF_RG8,
		PF_RG32F,
		PF_RG16F,
		PF_RGB8,
		PF_RGB32F,
		PF_RGB16F,
		PF_RGBA8,
		PF_RGBA16_UShort,
		PF_RGBA32F,
		PF_DepthComponent24,
		PF_DepthStencil,
		PF_ShadowDepth,
		PF_MAX = PF_ShadowDepth + 1,
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

	struct PixelFormatInfo {
		const WChar*    Name;
		int             Size,
		                Channels;
		// Is supported on the current platform
		bool            Supported;
		EPixelFormat    PixelFormat;
	};

	// Maps members of EPixelFormat to a PixelFormatInfo describing the platform specific format.
	extern const PixelFormatInfo PixelFormats[PF_MAX];

}