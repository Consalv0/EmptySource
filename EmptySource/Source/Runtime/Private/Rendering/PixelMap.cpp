
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/PixelMap.h"

namespace EmptySource {

	PixelMap::PixelMap()
		: Width(0), Height(0), Depth(0), PixelFormat(EColorFormat::CF_Red), Data(NULL) {
	}

	PixelMap::PixelMap(int Width, int Height, int Depth, EColorFormat PixelFormat)
		: Width(Width), Height(Height), Depth(Depth), PixelFormat(PixelFormat) {
		PixelMapUtility::CreateData(Width, Height, Depth, PixelFormat, Data);
	}

	PixelMap::PixelMap(const PixelMap & Other)
		: Width(Other.Width), Height(Other.Height), Depth(Other.Depth), PixelFormat(Other.PixelFormat) {
	}

	size_t PixelMap::GetMemorySize() const {
		return Width * Height * Depth * PixelMapUtility::PixelSize(PixelFormat);
	}

	PixelMap & PixelMap::operator=(const PixelMap & Other) {
		if (Data) {
			delete[] Data;
		}
		Width = Other.Width, Height = Other.Height; Depth = Other.Depth;
		PixelFormat = Other.PixelFormat;
		PixelMapUtility::CreateData(Width, Height, Depth, PixelFormat, Data);
		memcpy(Data, Other.Data, Width * Height * PixelMapUtility::PixelSize(PixelFormat));
		return *this;
	}

	const void * PixelMap::PointerToValue() const {
		return Data;
	}

	PixelMap::~PixelMap() {
		delete[] Data;
	}

	void PixelMapUtility::CreateData(int Width, int Height, int Depth, EColorFormat PixelFormat, void *& Data) {
		switch (PixelFormat) {
		case CF_Red:     Data = new UCharRed[Width * Height * Depth]; break;
		case CF_Red16F:  Data = new FloatRed[Width * Height * Depth]; break;
		case CF_Red32F:  Data = new FloatRed[Width * Height * Depth]; break;
		case CF_RG:      Data = new UCharRG[Width * Height * Depth]; break;
		case CF_RG16F:   Data = new FloatRG[Width * Height * Depth]; break;
		case CF_RGB:     Data = new UCharRGB[Width * Height * Depth]; break;
		case CF_RGB16F:  Data = new FloatRGB[Width * Height * Depth]; break;
		case CF_RGB32F:  Data = new FloatRGB[Width * Height * Depth]; break;
		case CF_RGBA:    Data = new UCharRGBA[Width * Height * Depth]; break;
		case CF_RGBA16F: Data = new FloatRGBA[Width * Height * Depth]; break;
		case CF_RGBA32F: Data = new FloatRGBA[Width * Height * Depth]; break;
		}
	}

	unsigned int PixelMapUtility::PixelSize(const EColorFormat & Format) {
		switch (Format) {
		case CF_Red:     return sizeof(UCharRed);
		case CF_Red16F:  return sizeof(FloatRed);
		case CF_Red32F:  return sizeof(FloatRed);
		case CF_RG:      return sizeof(UCharRG);
		case CF_RG16F:   return sizeof(FloatRG);
		case CF_RGB:     return sizeof(UCharRGB);
		case CF_RGB16F:  return sizeof(FloatRGB);
		case CF_RGB32F:  return sizeof(FloatRGB);
		case CF_RGBA:    return sizeof(UCharRGBA);
		case CF_RGBA16F: return sizeof(FloatRGBA);
		case CF_RGBA32F: return sizeof(FloatRGBA);
		default:
			return 0;
		}
	}

	unsigned int PixelMapUtility::PixelSize(const PixelMap & Map) {
		return PixelSize(Map.PixelFormat);
	}

	unsigned int PixelMapUtility::PixelChannels(const EColorFormat & Format) {
		switch (Format) {
		case CF_Red:     return 1u;
		case CF_Red16F:  return 1u;
		case CF_Red32F:  return 1u;
		case CF_RG:      return 2u;
		case CF_RG16F:   return 2u;
		case CF_RGB:     return 3u;
		case CF_RGB16F:  return 3u;
		case CF_RGB32F:  return 3u;
		case CF_RGBA:    return 4u;
		case CF_RGBA16F: return 4u;
		case CF_RGBA32F: return 4u;
		}
		return 0;
	}

	void PixelMapUtility::FlipVertically(PixelMap & Map) {
		switch (Map.PixelFormat) {
			case CF_Red:     _FlipVertically<UCharRed> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_Red16F:  _FlipVertically<FloatRed> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_Red32F:  _FlipVertically<FloatRed> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_RG:      _FlipVertically<UCharRG>  (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_RG16F:   _FlipVertically<FloatRG>  (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_RGB:     _FlipVertically<UCharRGB> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_RGB16F:  _FlipVertically<FloatRGB> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_RGB32F:  _FlipVertically<FloatRGB> (Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_RGBA:    _FlipVertically<UCharRGBA>(Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_RGBA16F: _FlipVertically<FloatRGBA>(Map.Width, Map.Height, Map.Depth, Map.Data); break;
			case CF_RGBA32F: _FlipVertically<FloatRGBA>(Map.Width, Map.Height, Map.Depth, Map.Data); break;
		}
	}

	unsigned char * PixelMapUtility::GetCharPixelAt(PixelMap & Map, const unsigned int & X, const unsigned int & Y, const unsigned int & Z) {
		return &((unsigned char *)Map.Data)[
			Z * Map.Width * Map.Height * PixelChannels(Map.PixelFormat) +
			Y * Map.Width * PixelChannels(Map.PixelFormat) +
			X * PixelChannels(Map.PixelFormat)
		];
	}

	float * PixelMapUtility::GetFloatPixelAt(PixelMap & Map, const unsigned int & X, const unsigned int & Y, const unsigned int & Z) {
		return &((float *)Map.Data)[
			Z * Map.Width * Map.Height * PixelChannels(Map.PixelFormat) +
			Y * Map.Width * PixelChannels(Map.PixelFormat) + 
			X * PixelChannels(Map.PixelFormat)
		];
	}

	unsigned char * PixelMapUtility::GetCharPixelAt(PixelMap & Map, const size_t & Index) {
		return &((unsigned char *)Map.Data)[Index * PixelChannels(Map.PixelFormat)];
	}

	float * PixelMapUtility::GetFloatPixelAt(PixelMap & Map, const size_t & Index) {
		return &((float *)Map.Data)[Index * PixelChannels(Map.PixelFormat)];
	}

	void PixelMapUtility::PerPixelOperator(PixelMap & Map, std::function<void(unsigned char *, const unsigned char &)> const & Function) {
		const unsigned char & Count = PixelChannels(Map.PixelFormat);
		for (unsigned int z = 0; z < Map.Depth; ++z) {
			for (unsigned int y = 0; y < Map.Height; ++y) {
				for (unsigned int x = 0; x < Map.Width; ++x) {
					Function(GetCharPixelAt(Map, x, y, z), Count);
				}
			}
		}
	}

	void PixelMapUtility::PerPixelOperator(PixelMap & Map, std::function<void(float *, const unsigned char &)> const & Function) {
		const unsigned char & Count = PixelChannels(Map.PixelFormat);
		for (unsigned int z = 0; z < Map.Depth; ++z) {
			for (unsigned int y = 0; y < Map.Height; ++y) {
				for (unsigned int x = 0; x < Map.Width; ++x) {
					Function(GetFloatPixelAt(Map, x, y, z), Count);
				}
			}
		}
	}

	bool PixelMapUtility::ColorFormatIsFloat(EColorFormat Format) {
		switch (Format) {
			case CF_Red16F:
			case CF_Red32F:
			case CF_RG16F:
			case CF_RGB16F:
			case CF_RGB32F:
			case CF_RGBA16F:
			case CF_RGBA32F:
				return true;
			case CF_Red:
			case CF_RG:
			case CF_RGB:
			case CF_RGBA:
			default:
				return false;
		}
	}

}