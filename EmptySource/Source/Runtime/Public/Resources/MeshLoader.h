#pragma once

#include "Files/FileManager.h"
#include "Core/Transform.h"
#include "Mesh/Mesh.h"

#include <queue>
#include <future>

namespace EmptySource {

	struct MeshLoader {
	public:
		struct FileData {
			const FileStream * File;
			bool Optimize;
			TArray<MeshData> Meshes;
			TArray<Transform> MeshTransforms;

			//* The model data has been loaded
			bool bLoaded;

			FileData(const FileStream * File, bool Optimize);
			FileData(const FileData & Other) = delete;
			FileData& operator = (FileData & Other);
		};

	private:
		typedef std::function<void(FileData &)> FinishTaskFunction;
		typedef std::function<std::future<bool>(FileData &)> FutureTask;

		static bool _TaskRunning;

		struct Task {
			FileData Data;
			FinishTaskFunction FinishFunction;
			std::function<std::future<bool>(FileData &)> Future;

			Task(const Task& Other) = delete;
			Task(const FileStream * File, bool Optimize, FinishTaskFunction FinishFunction, FutureTask Future);
		};

		static bool RecognizeFileExtensionAndLoad(FileData & Data);
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
		static bool Load(FileData & Data);
		static void LoadAsync(FileStream * File, bool Optimize, FinishTaskFunction OnComplete);
	};

}