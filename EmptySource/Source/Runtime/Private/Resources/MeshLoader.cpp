﻿
#include "Engine/Log.h"
#include "Engine/Core.h"
#include "Resources/OBJLoader.h"
#include "Resources/FBXLoader.h"
#include "Resources/MeshLoader.h"

#include <future>
#include <thread>

namespace EmptySource {

	bool MeshLoader::_TaskRunning;
	std::queue<MeshLoader::Task*> MeshLoader::PendingTasks = std::queue<Task*>();
	std::future<bool> MeshLoader::CurrentFuture;
	std::mutex QueueLock;

	bool MeshLoader::RecognizeFileExtensionAndLoad(FileData & Data) {
		const WString Extension = Data.File->GetExtension();
		if (Text::CompareIgnoreCase(Extension, WString(L"FBX"))) {
			return FBXLoader::Load(Data);
		}
		if (Text::CompareIgnoreCase(Extension, WString(L"OBJ"))) {
			return OBJLoader::Load(Data);
		}

		return false;
	}

	bool MeshLoader::Initialize() {
		if (std::thread::hardware_concurrency() <= 1) {
			LOG_CORE_WARN(L"The aviable cores ({:d}) are insuficient for asyncronus loaders", std::thread::hardware_concurrency());
			return false;
		}
		if (!FBXLoader::InitializeSdkManager()) {
			return false;
		}

		// Initialize with all cores/threads status are free to use
		// AsyncTask.

		return true;
	}

	void MeshLoader::UpdateStatus() {
		if (!PendingTasks.empty() && CurrentFuture.valid() && !_TaskRunning) {
			CurrentFuture.get();
			PendingTasks.front()->FinishFunction(PendingTasks.front()->Data);
			delete PendingTasks.front();
			PendingTasks.pop();
		}
		if (!PendingTasks.empty() && !CurrentFuture.valid() && !_TaskRunning) {
			CurrentFuture = PendingTasks.front()->Future(PendingTasks.front()->Data);
		}
	}

	void MeshLoader::FinishAsyncTasks() {
		do {
			FinishCurrentAsyncTask();
			UpdateStatus();
		} while (!PendingTasks.empty());
	}

	void MeshLoader::FinishCurrentAsyncTask() {
		if (!PendingTasks.empty() && CurrentFuture.valid()) {
			CurrentFuture.get();
			PendingTasks.front()->FinishFunction(PendingTasks.front()->Data);
			delete PendingTasks.front();
			PendingTasks.pop();
		}
	}

	size_t MeshLoader::GetAsyncTaskCount() {
		return PendingTasks.size();
	}

	void MeshLoader::Exit() {
		if (CurrentFuture.valid())
			CurrentFuture.get();
	}

	bool MeshLoader::Load(FileData & Data) {
		if (Data.File == NULL) return false;

		if (_TaskRunning) {
			FinishCurrentAsyncTask();
		}

		_TaskRunning = true;
		LOG_CORE_DEBUG(L"Reading File Model '{}'", Data.File->GetShortPath());
		Data.bLoaded = RecognizeFileExtensionAndLoad(Data);
		_TaskRunning = false;
		return Data.bLoaded;
	}

	void MeshLoader::LoadAsync(FileStream * File, bool Optimize, FinishTaskFunction Then) {
		if (File == NULL) return;

		PendingTasks.push(new Task{
			File, Optimize, Then,
			[](FileData & Data) -> std::future<bool> {
				std::future<bool> Task = std::async(std::launch::async, Load, std::ref(Data));
				return std::move(Task);
			}
			});
	}

	MeshLoader::FileData::FileData(const FileStream * File, bool Optimize) :
		File(File), Optimize(Optimize), Meshes(), MeshTransforms(), bLoaded() {
	}

	MeshLoader::FileData & MeshLoader::FileData::operator=(FileData & Other) {
		File = Other.File;
		Optimize = Other.Optimize;
		Meshes.swap(Other.Meshes);
		MeshTransforms.swap(Other.MeshTransforms);
		bLoaded = Other.bLoaded;
		return *this;
	}

	MeshLoader::Task::Task(const FileStream * File, bool Optimize, FinishTaskFunction FinishFunction, FutureTask Future) :
		Data(File, Optimize), FinishFunction(FinishFunction), Future(Future) {
	}

}