#pragma once

#include "CoreMinimal.h"
#include "Files/FileManager.h"
#include "Rendering/Bitmap.h"

namespace EmptySource {

	class ImageConversion {
	public:
		template<typename T>
		static bool LoadFromFile(Bitmap<T> & RefBitmap, FileStream * File, bool FlipVertically = true);

		static int GetChannelCount(FileStream * File);

		static bool IsHDR(FileStream * File);

		static EColorFormat GetColorFormat(FileStream * File);

		static bool EncodeToFile(const Bitmap<FloatRed> & RefBitmap, FileStream * File);
	};

}