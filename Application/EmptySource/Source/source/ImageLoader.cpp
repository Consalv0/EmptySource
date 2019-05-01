
#include "../include/ImageLoader.h"

// --- Visual Studio
#if defined(_MSC_VER) && (_MSC_VER >= 1310) 
#pragma warning( disable : 4996 ) /*VS does not like fopen, but fopen_s is not standard C so unusable here*/
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "External/STB/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/STB/stb_image_write.h"

#define _LoadImage(BitmapFormat, STBIFormat) \
	if (File == NULL) return false; \
	int Width, Height, Comp; \
	FILE * FILEFile = fopen(WStringToString(File->GetPath()).c_str(), "rb"); \
	unsigned char * Image = stbi_load_from_file(FILEFile, &Width, &Height, &Comp, STBIFormat); \
	if (Image == NULL) return false; \
	RefBitmap = Bitmap<BitmapFormat>(Width, Height); \
	memmove(&RefBitmap[0], &Image[0], Width * Height * sizeof(BitmapFormat)); \
	return true;

#define _LoadImagef(BitmapFormat, STBIFormat) \
	if (File == NULL) return false; \
	int Width, Height, Comp; \
	FILE * FILEFile = fopen(WStringToString(File->GetPath()).c_str(), "rb"); \
	float * Image = stbi_loadf_from_file(FILEFile, &Width, &Height, &Comp, STBIFormat); \
	if (Image == NULL) return false; \
	RefBitmap = Bitmap<BitmapFormat>(Width, Height); \
	memmove(&RefBitmap[0], &Image[0], Width * Height * sizeof(BitmapFormat)); \
	return true;

bool ImageLoader::Load(Bitmap<UCharRGBA>& RefBitmap, FileStream * File) {
	_LoadImage(UCharRGBA, STBI_rgb_alpha);
}

bool ImageLoader::Load(Bitmap<UCharRGB>& RefBitmap, FileStream * File) {
	_LoadImage(UCharRGB, STBI_rgb);
}

bool ImageLoader::Load(Bitmap<UCharRG>& RefBitmap, FileStream * File) {
	_LoadImage(UCharRG, STBI_grey_alpha);
}

bool ImageLoader::Load(Bitmap<UCharRed>& RefBitmap, FileStream * File) {
	_LoadImage(UCharRed, STBI_grey);
}

bool ImageLoader::Load(Bitmap<FloatRGB>& RefBitmap, FileStream * File) {
	_LoadImagef(FloatRGB, STBI_rgb);
}

bool ImageLoader::Write(const Bitmap<FloatRed>& RefBitmap, FileStream * File) {
	// TArray<unsigned char> Pixels(RefBitmap.GetWidth() * RefBitmap.GetHeight());
	// TArray<unsigned char>::iterator it = Pixels.begin();
	// for (int y = RefBitmap.GetHeight() - 1; y >= 0; --y)
	// 	for (int x = 0; x < RefBitmap.GetWidth(); ++x)
	// 		*it++ = Math::Clamp(int(RefBitmap(x, y) * 0x100), 0xff);
	// return !lodepng::encode(WStringToString(File->GetPath()), Pixels, RefBitmap.GetWidth(), RefBitmap.GetHeight(), LCT_GREY);
	return false;
}
