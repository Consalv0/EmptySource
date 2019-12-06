
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

namespace ESource {

	bool ImageConversion::_TaskRunning;
	std::queue<ImageConversion::Task*> ImageConversion::PendingTasks = std::queue<ImageConversion::Task*>();
	std::future<bool> ImageConversion::CurrentFuture;
	std::mutex ImageConversionQueueLock;

	void ImageConversion::FinishCurrentAsyncTask() {
		if (!PendingTasks.empty() && CurrentFuture.valid()) {
			CurrentFuture.get();
			PendingTasks.front()->FinishFunction(PendingTasks.front()->Info);
			delete PendingTasks.front();
			PendingTasks.pop();
		}
	}

	bool ImageConversion::Initialize() {
		if (std::thread::hardware_concurrency() <= 1) {
			LOG_CORE_WARN(L"The aviable cores ({:d}) are insuficient for asyncronus loaders", std::thread::hardware_concurrency());
			return false;
		}
		return true;
	}

	void ImageConversion::UpdateStatus() {
		if (!PendingTasks.empty() && CurrentFuture.valid() && !_TaskRunning) {
			CurrentFuture.get();
			PendingTasks.front()->FinishFunction(PendingTasks.front()->Info);
			delete PendingTasks.front();
			PendingTasks.pop();
		}
		if (!PendingTasks.empty() && !CurrentFuture.valid() && !_TaskRunning) {
			CurrentFuture = PendingTasks.front()->Future(PendingTasks.front()->Info, PendingTasks.front()->Options);
		}
	}

	void ImageConversion::FinishAsyncTasks() {
		do {
			FinishCurrentAsyncTask();
			UpdateStatus();
		} while (!PendingTasks.empty());
	}

	size_t ImageConversion::GetAsyncTaskCount() {
		return PendingTasks.size();
	}

	void ImageConversion::Exit() {
		if (CurrentFuture.valid())
			CurrentFuture.get();
	}

	bool ImageConversion::Load(PixelMapInfo & Data, const ParsingOptions & Options) {
		if (Options.File == NULL) return false;
		int Width, Height, Comp;
		stbi_set_flip_vertically_on_load(Options.FlipVertically);
		FILE * FILEFile = fopen(Text::WideToNarrow(Options.File->GetPath()).c_str(), "rb");
		void * Image = NULL; 
		if (PixelMapUtility::FormatIsFloat(Options.Format))
			Image = stbi_loadf_from_file(FILEFile, &Width, &Height, &Comp, PixelFormats[Options.Format].Channels);
		else
			Image = stbi_load_from_file(FILEFile, &Width, &Height, &Comp, PixelFormats[Options.Format].Channels);
		fclose(FILEFile);
		if (Image == NULL) {
			LOG_CORE_ERROR(L"Texture '{0}' coldn´t be loaded", Options.File->GetFileName().c_str());
			return false;
		}
		Data.Pixels.SetData(Width, Height, 1, Options.Format, Image);
		stbi_image_free(Image);
		return true;
	}

	void ImageConversion::LoadAsync(const ParsingOptions & Options, FinishTaskFunction Then)
	{
		if (Options.File == NULL) return;

		PendingTasks.push(
			new Task{ Options, Then, [](PixelMapInfo & Data, const ParsingOptions & Options) -> std::future<bool> {
				std::future<bool> Task = std::async(std::launch::async, Load, std::ref(Data), std::ref(Options));
					return std::move(Task);
				}
			}
		);
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
		// for (int Y = RefBitmap.GetHeight() - 1; Y >= 0; --Y)
		// 	for (int X = 0; X < RefBitmap.GetWidth(); ++X)
		// 		*it++ = Math::Clamp(int(RefBitmap(X, Y) * 0x100), 0xff);
		// return !lodepng::encode(WStringToString(File->GetPath()), Pixels, RefBitmap.GetWidth(), RefBitmap.GetHeight(), LCT_GREY);
		return false;
	}

	void ImageConversion::PixelMapInfo::Transfer(PixelMapInfo & Other) {
		Pixels = Other.Pixels;
		bSuccess = Other.bSuccess;
	}

	ImageConversion::PixelMapInfo::PixelMapInfo()
		: Pixels(), bSuccess(false) {
	}

	ImageConversion::Task::Task(const ParsingOptions & Options, FinishTaskFunction FinishFunction, FutureTask Future) :
		Info(), Options(Options), FinishFunction(FinishFunction), Future(Future) {
	}

}