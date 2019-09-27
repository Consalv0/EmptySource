
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

	bool ImageConversion::LoadFromFile(PixelMap& RefBitmap, FileStream * File, EPixelFormat Format, bool FlipVertically) {
		if (File == NULL) return false;
		int Width, Height, Comp;
		stbi_set_flip_vertically_on_load(FlipVertically);
		FILE * FILEFile = fopen(Text::WideToNarrow(File->GetPath()).c_str(), "rb");
		void * Image = NULL; 
		if (PixelMapUtility::FormatIsFloat(Format))
			Image = stbi_loadf_from_file(FILEFile, &Width, &Height, &Comp, PixelFormats[Format].Channels);
		else
			Image = stbi_load_from_file(FILEFile, &Width, &Height, &Comp, PixelFormats[Format].Channels);
		fclose(FILEFile);
		if (Image == NULL) {
			LOG_CORE_ERROR(L"Texture '{0}' coldn´t be loaded", File->GetFileName().c_str());
			return false;
		}
		RefBitmap.SetData(Width, Height, 1, Format, Image);
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

	EPixelFormat ImageConversion::GetColorFormat(FileStream * File) {
		EPixelFormat InputColorFormat = EPixelFormat::PF_Unknown;
		bool IsFloat32 = ImageConversion::IsHDR(File);

		switch (ImageConversion::GetChannelCount(File)) {
		case 1:
			if (IsFloat32) InputColorFormat = EPixelFormat::PF_R32F;
			else InputColorFormat = EPixelFormat::PF_R8;
		case 2:
			if (IsFloat32) InputColorFormat = EPixelFormat::PF_RG32F;
			else InputColorFormat = EPixelFormat::PF_RG8;
		case 3:
			if (IsFloat32) InputColorFormat = EPixelFormat::PF_RGB32F;
			else InputColorFormat = EPixelFormat::PF_RGB8;
		case 4:
			if (IsFloat32) InputColorFormat = EPixelFormat::PF_RGBA32F;
			else InputColorFormat = EPixelFormat::PF_RGBA8;
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