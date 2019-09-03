
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Resources/ImageConversion.h"

// --- Visual Studio
#if defined(_MSC_VER) && (_MSC_VER >= 1310) 
#pragma warning( disable : 4996 ) /*VS does not like fopen, but fopen_s is not standard C so unusable here*/
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace EmptySource {

	template<typename T>
	struct LoadFromFile {
		static T * Load(FILE *File, int *Width, int *Height, int *Comp, int Channels) = 0;
	};

	template<>
	struct LoadFromFile<unsigned char> {
		static unsigned char * Load(FILE *File, int *Width, int *Height, int *Comp, int Channels) {
			return stbi_load_from_file(File, Width, Height, Comp, Channels);
		}
	};

	template<>
	struct LoadFromFile<float> {
		static float * Load(FILE *File, int *Width, int *Height, int *Comp, int Channels) {
			return stbi_loadf_from_file(File, Width, Height, Comp, Channels);
		}
	};

	template<typename T>
	bool _LoadBitmap(Bitmap<T>& RefBitmap, FileStream * File, bool FlipVertically = true) {
		if (File == NULL) return false;
		int Width, Height, Comp;
		stbi_set_flip_vertically_on_load(FlipVertically);
		FILE * FILEFile = fopen(Text::WideToNarrow(File->GetPath()).c_str(), "rb");
		auto * Image = LoadFromFile<typename T::Range>::Load(FILEFile, &Width, &Height, &Comp, T::Channels);
		fclose(FILEFile);
		if (Image == NULL) {
			LOG_CORE_ERROR(L"Texture '{0}' coldn´t be loaded", File->GetFileName().c_str());
			return false;
		}
		RefBitmap = Bitmap<T>(Width, Height);
		memmove(&RefBitmap[0], &Image[0], Width * Height * sizeof(T));
		return true;
	}

	template<>
	bool ImageConversion::LoadFromFile(Bitmap<UCharRGBA>& RefBitmap, FileStream * File, bool FlipVertically) {
		return _LoadBitmap<UCharRGBA>(RefBitmap, File, FlipVertically);
	}

	template<>
	bool ImageConversion::LoadFromFile(Bitmap<UCharRGB>& RefBitmap, FileStream * File, bool FlipVertically) {
		return _LoadBitmap<UCharRGB>(RefBitmap, File, FlipVertically);
	}

	template<>
	bool ImageConversion::LoadFromFile(Bitmap<UCharRG>& RefBitmap, FileStream * File, bool FlipVertically) {
		return _LoadBitmap<UCharRG>(RefBitmap, File, FlipVertically);
	}

	template<>
	bool ImageConversion::LoadFromFile(Bitmap<UCharRed>& RefBitmap, FileStream * File, bool FlipVertically) {
		return _LoadBitmap<UCharRed>(RefBitmap, File, FlipVertically);
	}

	template<>
	bool ImageConversion::LoadFromFile(Bitmap<FloatRGB>& RefBitmap, FileStream * File, bool FlipVertically) {
		return _LoadBitmap<FloatRGB>(RefBitmap, File, FlipVertically);
	}

	template<>
	bool ImageConversion::LoadFromFile(Bitmap<FloatRGBA>& RefBitmap, FileStream * File, bool FlipVertically) {
		return _LoadBitmap<FloatRGBA>(RefBitmap, File, FlipVertically);
	}

	int ImageConversion::GetChannelCount(FileStream * File) {
		if (File == NULL) return 0;
		int Width, Height, Comp;
		FILE * FILEFile = fopen(Text::WideToNarrow(File->GetPath()).c_str(), "rb");
		stbi_info_from_file(FILEFile, &Width, &Height, &Comp);
		return Comp;
	}

	bool ImageConversion::IsHDR(FileStream * File) {
		if (File == NULL) return false;
		return stbi_is_hdr(Text::WideToNarrow(File->GetPath()).c_str());
	}

	EColorFormat ImageConversion::GetColorFormat(FileStream * File) {
		EColorFormat InputColorFormat = EColorFormat::CF_RGBA;
		bool IsFloat32 = ImageConversion::IsHDR(File);

		switch (ImageConversion::GetChannelCount(File)) {
		case 1:
			if (IsFloat32) { ES_CORE_ASSERT(true, "Color format not supported"); }
			else InputColorFormat = EColorFormat::CF_Red;
		case 2:
			if (IsFloat32) { ES_CORE_ASSERT(true, "Color format not supported"); }
			else InputColorFormat = EColorFormat::CF_RG;
		case 3:
			if (IsFloat32) InputColorFormat = EColorFormat::CF_RGB32F;
			else InputColorFormat = EColorFormat::CF_RGB;
		case 4:
			if (IsFloat32) InputColorFormat = EColorFormat::CF_RGBA32F;
			else InputColorFormat = EColorFormat::CF_RGBA;
		}

		return InputColorFormat;
	}

	bool ImageConversion::EncodeToFile(const Bitmap<FloatRed>& RefBitmap, FileStream * File) {
		// TArray<unsigned char> Pixels(RefBitmap.GetWidth() * RefBitmap.GetHeight());
		// TArray<unsigned char>::iterator it = Pixels.begin();
		// for (int y = RefBitmap.GetHeight() - 1; y >= 0; --y)
		// 	for (int x = 0; x < RefBitmap.GetWidth(); ++x)
		// 		*it++ = Math::Clamp(int(RefBitmap(x, y) * 0x100), 0xff);
		// return !lodepng::encode(WStringToString(File->GetPath()), Pixels, RefBitmap.GetWidth(), RefBitmap.GetHeight(), LCT_GREY);
		return false;
	}

}