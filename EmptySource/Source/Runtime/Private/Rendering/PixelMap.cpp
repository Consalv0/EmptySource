
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/PixelMap.h"

namespace ESource {

	PixelMap::PixelMap()
		: Width(0), Height(0), Depth(0), PixelFormat(PF_Unknown), Data(NULL) {
	}

	PixelMap::PixelMap(int Width, int Height, int Depth, EPixelFormat PixelFormat)
		: Width(Width), Height(Height), Depth(Depth), PixelFormat(PixelFormat), Data(NULL) {
		PixelMapUtility::CreateData(Width, Height, Depth, PixelFormat, Data);
	}

	PixelMap::PixelMap(int Width, int Height, int Depth, EPixelFormat PixelFormat, void *& InData)
		: Width(Width), Height(Height), Depth(Depth), PixelFormat(PixelFormat), Data(NULL) {
		PixelMapUtility::CreateData(Width, Height, Depth, PixelFormat, Data, InData);
	}

	PixelMap::PixelMap(const PixelMap & Other)
		: Width(Other.Width), Height(Other.Height), Depth(Other.Depth), PixelFormat(Other.PixelFormat), Data(NULL) {
		PixelMapUtility::CreateData(Width, Height, Depth, PixelFormat, Data, Other.Data);
	}

	void PixelMap::SetData(int InWidth, int InHeight, int InDepth, EPixelFormat InPixelFormat, void *& InData) {
		if (Data) {
			delete[] Data;
			Data = NULL;
		}
		Width = InWidth, Height = InHeight; Depth = InDepth;
		PixelFormat = InPixelFormat;
		PixelMapUtility::CreateData(Width, Height, Depth, PixelFormat, Data, InData);
	}

	size_t PixelMap::GetMemorySize() const {
		return Width * Height * Depth * PixelFormats[PixelFormat].Size;
	}

	PixelMap & PixelMap::operator=(const PixelMap & Other) {
		if (Data) {
			delete[] Data;
			Data = NULL;
		}
		Width = Other.Width, Height = Other.Height; Depth = Other.Depth;
		PixelFormat = Other.PixelFormat;
		PixelMapUtility::CreateData(Width, Height, Depth, PixelFormat, Data, Other.Data);
		return *this;
	}

	const void * PixelMap::PointerToValue() const {
		return Data;
	}

	PixelMap::~PixelMap() {
		delete[] Data;
	}

	void PixelMapUtility::CreateData(int Width, int Height, int Depth, EPixelFormat PixelFormat, void *& Data) {
		if (Data != NULL) return;
		const size_t Size = Width * Height * Depth * (size_t)PixelFormats[PixelFormat].Size;
		if (Size == 0) {
			Data = NULL;
			return;
		}
		if (FormatIsFloat(PixelFormat))
			Data = new float[Size];
		else if (FormatIsShort(PixelFormat))
			Data = new unsigned short[Size];
		else
			Data = new unsigned char[Size];
	}

	void PixelMapUtility::CreateData(int Width, int Height, int Depth, EPixelFormat PixelFormat, void *& Target, void * Data) {
		CreateData(Width, Height, Depth, PixelFormat, Target);
		if (Target == NULL || Data == NULL) return;
		memcpy(Target, Data, Width * Height * Depth * (size_t)PixelFormats[PixelFormat].Size);
	}

	void PixelMapUtility::FlipVertically(PixelMap & Map) {
		switch (Map.PixelFormat) {
			case PF_R8:      _FlipVertically<UCharRed> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case PF_R32F:    _FlipVertically<FloatRed> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case PF_RG8:     _FlipVertically<UCharRG>  (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case PF_RG32F:   _FlipVertically<FloatRG>  (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case PF_RGB8:    _FlipVertically<UCharRGB> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case PF_RGB32F:  _FlipVertically<FloatRGB> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case PF_RGBA8:   _FlipVertically<UCharRGBA>(Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case PF_RGBA32F: _FlipVertically<FloatRGBA>(Map.Width, Map.Height, Map.Depth, Map.Data); break;
		}
	}

	unsigned char * PixelMapUtility::GetCharPixelAt(PixelMap & Map, const unsigned int & X, const unsigned int & Y, const unsigned int & Z) {
		const int Channels = PixelFormats[Map.PixelFormat].Channels;
		return &((unsigned char *)Map.Data)[
			Z * Map.Width * Map.Height * Channels +
			Y * Map.Width * Channels +
			X * Channels
		];
	}

	float * PixelMapUtility::GetFloatPixelAt(PixelMap & Map, const unsigned int & X, const unsigned int & Y, const unsigned int & Z) {
		const int Channels = PixelFormats[Map.PixelFormat].Channels;
		return &((float *)Map.Data)[
			Z * Map.Width * Map.Height * Channels +
			Y * Map.Width * Channels +
			X * Channels
		];
	}

	unsigned char * PixelMapUtility::GetCharPixelAt(PixelMap & Map, const size_t & Index) {
		return &((unsigned char *)Map.Data)[Index * PixelFormats[Map.PixelFormat].Channels];
	}

	float * PixelMapUtility::GetFloatPixelAt(PixelMap & Map, const size_t & Index) {
		return &((float *)Map.Data)[Index * PixelFormats[Map.PixelFormat].Channels];
	}

	void PixelMapUtility::PerPixelOperator(PixelMap & Map, std::function<void(unsigned char *, const unsigned char &)> const & Function) {
		const unsigned char & Channels = (unsigned char)PixelFormats[Map.PixelFormat].Channels;
		for (unsigned int z = 0; z < Map.Depth; ++z) {
			for (unsigned int y = 0; y < Map.Height; ++y) {
				for (unsigned int x = 0; x < Map.Width; ++x) {
					Function(GetCharPixelAt(Map, x, y, z), Channels);
				}
			}
		}
	}

	void PixelMapUtility::PerPixelOperator(PixelMap & Map, std::function<void(float *, const unsigned char &)> const & Function) {
		const unsigned char & Channels = (unsigned char)PixelFormats[Map.PixelFormat].Channels;
		for (unsigned int z = 0; z < Map.Depth; ++z) {
			for (unsigned int y = 0; y < Map.Height; ++y) {
				for (unsigned int x = 0; x < Map.Width; ++x) {
					Function(GetFloatPixelAt(Map, x, y, z), Channels);
				}
			}
		}
	}

	bool PixelMapUtility::FormatIsFloat(EPixelFormat Format) {
		switch (Format) {
		case PF_R32F:
		case PF_RG32F:
		case PF_RG16F:
		case PF_RGB32F:
		case PF_RGBA32F:
		case PF_DepthStencil:
		case PF_ShadowDepth:
			return true;
		default:
			return false;
		}
	}

	bool PixelMapUtility::FormatIsShort(EPixelFormat Format) {
		switch (Format) {
		case PF_RGBA16_UShort:
			return true;
		default:
			return false;
		}
	}

}