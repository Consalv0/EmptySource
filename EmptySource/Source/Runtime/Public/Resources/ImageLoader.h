#pragma once

#include "Engine/Core.h"
#include "Files/FileManager.h"
#include "Graphics/Bitmap.h"

namespace EmptySource {

	class ImageLoader {
	public:
		template<typename T>
		static bool Load(Bitmap<T> & RefBitmap, FileStream * File, bool FlipVertically = true);

		static bool Write(const Bitmap<FloatRed> & RefBitmap, FileStream * File);
	};

}