#pragma once

#include "../include/FileManager.h"
#include "../include/Bitmap.h"

class ImageLoader {
public:
	static bool Load(Bitmap<UCharRGBA> & RefBitmap, FileStream * File);
	static bool Load(Bitmap<UCharRGB> & RefBitmap, FileStream * File);
	static bool Load(Bitmap<UCharRG> & RefBitmap, FileStream * File);
	static bool Load(Bitmap<UCharRed> & RefBitmap, FileStream * File);
	static bool Write(const Bitmap<FloatRed> & RefBitmap, FileStream * File);
};