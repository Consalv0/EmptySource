#pragma once

#include "Files/FileManager.h"
#include "Core/Transform.h"
#include "Rendering/Mesh.h"

#include <queue>
#include <future>

namespace ESource {

	class MeshParser {
	public:
		struct ResourceData {
			const FileStream * File;
			bool Optimize;
			TArray<MeshData> Meshes;
			TArray<Transform> MeshTransforms;

			//* The model data has been loaded
			bool bLoaded;

			ResourceData(const FileStream * File, bool Optimize);
			ResourceData(const ResourceData & Other) = delete;
			ResourceData& operator = (ResourceData & Other);
		};

	private:
		typedef std::function<void(ResourceData &)> FinishTaskFunction;
		typedef std::function<std::future<bool>(ResourceData &)> FutureTask;

		static bool _TaskRunning;

		struct Task {
			ResourceData Data;
			FinishTaskFunction FinishFunction;
			std::function<std::future<bool>(ResourceData &)> Future;

			Task(const Task& Other) = delete;
			Task(const FileStream * File, bool Optimize, FinishTaskFunction FinishFunction, FutureTask Future);
		};

		static bool RecognizeFileExtensionAndLoad(ResourceData & Data);

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

		static bool Load(ResourceData & Data);

		static void LoadAsync(FileStream * File, bool Optimize, FinishTaskFunction OnComplete);

	};

}