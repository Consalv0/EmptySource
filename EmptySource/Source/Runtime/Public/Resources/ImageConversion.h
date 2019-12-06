#pragma once

#include "CoreMinimal.h"
#include "Files/FileManager.h"
#include "Rendering/PixelMap.h"

#include <future>

namespace ESource {

	class ImageConversion {
	public:
		struct ParsingOptions {
			const FileStream * File;
			EPixelFormat Format;
			bool FlipVertically;
		};

		struct PixelMapInfo {
			PixelMap Pixels;

			bool bSuccess;

			void Transfer(PixelMapInfo & Other);

			PixelMapInfo();

			PixelMapInfo(const PixelMapInfo & Other) = delete;
		};

	private:
		typedef std::function<void(PixelMapInfo &)> FinishTaskFunction;
		typedef std::function<std::future<bool>(PixelMapInfo &, const ParsingOptions &)> FutureTask;

		static bool _TaskRunning;

		struct Task {
			ParsingOptions Options;
			PixelMapInfo Info;
			FinishTaskFunction FinishFunction;
			FutureTask Future;

			Task(const Task& Other) = delete;
			Task(const ParsingOptions & Options, FinishTaskFunction FinishFunction, FutureTask Future);
		};

		static void FinishCurrentAsyncTask();

		//* Mesh Loading Threads
		static std::queue<Task *> PendingTasks;
		static std::future<bool> CurrentFuture;

	public:
		static bool Initialize();

		static void UpdateStatus();

		static void FinishAsyncTasks();

		static size_t GetAsyncTaskCount();

		static void Exit();

		static bool Load(PixelMapInfo & Data, const ParsingOptions & Options);

		static void LoadAsync(const ParsingOptions & Options, FinishTaskFunction OnComplete);

		static int GetChannelCount(FileStream * File);

		static bool IsHDR(FileStream * File);

		static EPixelFormat GetColorFormat(FileStream * File);

		static bool EncodeToFile(const PixelMap & RefBitmap, FileStream * File);
	};

}