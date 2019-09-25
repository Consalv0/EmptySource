
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

	bool ImageConversion::LoadFromFile(PixelMap& RefBitmap, FileStream * File, EColorFormat Format, bool FlipVertically) {
		if (File == NULL) return false;
		int Width, Height, Comp;
		stbi_set_flip_vertically_on_load(FlipVertically);
		FILE * FILEFile = fopen(Text::WideToNarrow(File->GetPath()).c_str(), "rb");
		void * Image = NULL; 
		if (PixelMapUtility::ColorFormatIsFloat(Format))
			Image = stbi_loadf_from_file(FILEFile, &Width, &Height, &Comp, PixelMapUtility::PixelChannels(Format));
		else
			Image = stbi_load_from_file(FILEFile, &Width, &Height, &Comp, PixelMapUtility::PixelChannels(Format));
		fclose(FILEFile);
		if (Image == NULL) {
			LOG_CORE_ERROR(L"Texture '{0}' coldn´t be loaded", File->GetFileName().c_str());
			return false;
		}
		RefBitmap = PixelMap(Width, Height, 1, Format);
		memcpy((void *)RefBitmap.PointerToValue(), Image, Width * Height * PixelMapUtility::PixelSize(Format));
		stbi_image_free(Image);
		return true;
	}

	int ImageConversion::GetChannelCount(FileStream * File) {
		if (File == NULL) return 0;
		int Width, Height, Comp = 0;
		FILE * FILEFile = fopen(Text::WideToNarrow(File->GetPath()).c_str(), "rb");
		if (FILEFile) {
			stbi_info_from_file(FILEFile, &Width, &Height, &Comp);
			fclose(FILEFile);
		}
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

	bool ImageConversion::EncodeToFile(const PixelMap& RefBitmap, FileStream * File) {
		// TArray<unsigned char> Pixels(RefBitmap.GetWidth() * RefBitmap.GetHeight());
		// TArray<unsigned char>::iterator it = Pixels.begin();
		// for (int y = RefBitmap.GetHeight() - 1; y >= 0; --y)
		// 	for (int x = 0; x < RefBitmap.GetWidth(); ++x)
		// 		*it++ = Math::Clamp(int(RefBitmap(x, y) * 0x100), 0xff);
		// return !lodepng::encode(WStringToString(File->GetPath()), Pixels, RefBitmap.GetWidth(), RefBitmap.GetHeight(), LCT_GREY);
		return false;
	}

}