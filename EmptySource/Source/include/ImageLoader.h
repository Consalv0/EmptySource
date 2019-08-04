#pragma once

#include "../include/Core.h"
#include "../include/FileManager.h"
#include "../include/Bitmap.h"

namespace EmptySource {

	class ImageLoader {
	public:
		template<typename T>
		static bool Load(Bitmap<T> & RefBitmap, FileStream * File, bool FlipVertically = true);

		static bool Write(const Bitmap<FloatRed> & RefBitmap, FileStream * File);
	};

}