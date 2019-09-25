#pragma once

#include "CoreMinimal.h"
#include "Files/FileManager.h"
#include "Rendering/PixelMap.h"

namespace EmptySource {

	class ImageConversion {
	public:
		static bool LoadFromFile(PixelMap & RefBitmap, FileStream * File, EColorFormat Format, bool FlipVertically = true);

		static int GetChannelCount(FileStream * File);

		static bool IsHDR(FileStream * File);

		static EColorFormat GetColorFormat(FileStream * File);

		static bool EncodeToFile(const PixelMap & RefBitmap, FileStream * File);
	};

}