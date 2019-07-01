#pragma once

#include "../include/FileManager.h"
#include "../include/Transform.h"
#include "../include/Mesh.h"

#include <queue>
#include <future>

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
	struct Task {
		FileData Data;
		FinishTaskFunction FinishFunction;
		std::function<std::future<bool>(FileData &)> Future;

		Task(const Task& Other);
		Task(const FileStream * File, bool Optimize, FinishTaskFunction FinishFunction, FutureTask Future);
	};

	static bool RecognizeFileExtensionAndLoad(FileData & Data);

	//* Mesh Loading Threads
	static std::queue<Task *> PendingTasks;
	static std::future<bool> CurrentFuture;

public:
	static bool _TaskRunning;

	static bool Initialize();
	static void UpdateStatus();

	static void Exit();
	static bool Load(FileData & Data);
	static void LoadAsync(FileStream * File, bool Optimize, FinishTaskFunction OnComplete);
};