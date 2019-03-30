
#include "../include/Math/CoreMath.h"
#include "../include/ImageLoader.h"
#include "../include/Core.h"

#include "External/LodePNG/lodepng.h"
// #include "External/LodePNG/lodepng_util.h"

bool JPEGLoader::Load(const Bitmap<FloatRed>& RefBitmap, FileStream* File) {
	return false;
}

bool PNGLoader::Load(Bitmap<UCharRGBA>& RefBitmap, FileStream * File) {
	TArray<unsigned char> Image;
	unsigned int Width, Height;
	lodepng::State State;
	lodepng::decode(Image, Width, Height, WStringToString(File->GetPath()), LodePNGColorType::LCT_RGBA);
	
	RefBitmap = Bitmap<UCharRGBA>(Width, Height);
	memmove( &RefBitmap[0], &Image[0], Width * Height * sizeof(UCharRGBA) );

	return true;
}

bool PNGLoader::Load(Bitmap<UCharRGB>& RefBitmap, FileStream * File) {
	TArray<unsigned char> Image;
	unsigned int Width, Height;
	lodepng::State State;
	lodepng::decode(Image, Width, Height, WStringToString(File->GetPath()), LodePNGColorType::LCT_RGB);

	RefBitmap = Bitmap<UCharRGB>(Width, Height);
	memmove(&RefBitmap[0], &Image[0], Width * Height * sizeof(UCharRGB));

	return true;
}

bool PNGLoader::Write(const Bitmap<FloatRed>& RefBitmap, FileStream * File) {
	TArray<unsigned char> Pixels(RefBitmap.GetWidth() * RefBitmap.GetHeight());
	TArray<unsigned char>::iterator it = Pixels.begin();
	for (int y = RefBitmap.GetHeight() - 1; y >= 0; --y)
		for (int x = 0; x < RefBitmap.GetWidth(); ++x)
			*it++ = Math::Clamp(int(RefBitmap(x, y) * 0x100), 0xff);
	return !lodepng::encode(WStringToString(File->GetPath()), Pixels, RefBitmap.GetWidth(), RefBitmap.GetHeight(), LCT_GREY);
}
