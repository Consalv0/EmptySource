#pragma once

#include "../include/FileManager.h"
#include "../include/Bitmap.h"

class PNGLoader {
public:
	static bool Load(Bitmap<UCharRGBA> & RefBitmap, FileStream * File);
	static bool Load(Bitmap<UCharRGB> & RefBitmap, FileStream * File);
	static bool Write(const Bitmap<float> & RefBitmap, FileStream * File);
};

class JPEGLoader {
public:
	static bool Load(const Bitmap<float> & RefBitmap, FileStream * File);
	static bool Write(const Bitmap<float> & RefBitmap, FileStream * File);
};