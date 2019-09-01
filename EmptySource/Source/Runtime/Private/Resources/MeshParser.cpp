
#include "CoreMinimal.h"
#include "Rendering/Mesh.h"
#include "Resources/MeshParser.h"
#include "Resources/OBJLoader.h"
#include "Resources/FBXLoader.h"

#include <future>
#include <thread>

namespace EmptySource {

	bool MeshParser::_TaskRunning;
	std::queue<MeshParser::Task*> MeshParser::PendingTasks = std::queue<Task*>();
	std::future<bool> MeshParser::CurrentFuture;
	std::mutex QueueLock;

	bool MeshParser::RecognizeFileExtensionAndLoad(ResourceData & Data) {
		const WString Extension = Data.File->GetExtension();
		if (Text::CompareIgnoreCase(Extension, WString(L"FBX"))) {
			return FBXLoader::Load(Data);
		}
		if (Text::CompareIgnoreCase(Extension, WString(L"OBJ"))) {
			return OBJLoader::Load(Data);
		}

		return false;
	}

	bool MeshParser::Initialize() {
		if (std::thread::hardware_concurrency() <= 1) {
			LOG_CORE_WARN(L"The aviable cores ({:d}) are insuficient for asyncronus loaders", std::thread::hardware_concurrency());
			return false;
		}
		if (!FBXLoader::InitializeSdkManager()) {
			return false;
		}

		return true;
	}

	void MeshParser::UpdateStatus() {
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

	void MeshParser::FinishAsyncTasks() {
		do {
			FinishCurrentAsyncTask();
			UpdateStatus();
		} while (!PendingTasks.empty());
	}

	void MeshParser::FinishCurrentAsyncTask() {
		if (!PendingTasks.empty() && CurrentFuture.valid()) {
			CurrentFuture.get();
			PendingTasks.front()->FinishFunction(PendingTasks.front()->Data);
			delete PendingTasks.front();
			PendingTasks.pop();
		}
	}

	size_t MeshParser::GetAsyncTaskCount() {
		return PendingTasks.size();
	}

	void MeshParser::Exit() {
		if (CurrentFuture.valid())
			CurrentFuture.get();
	}

	bool MeshParser::Load(ResourceData & Data) {
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

	void MeshParser::LoadAsync(FileStream * File, bool Optimize, FinishTaskFunction Then) {
		if (File == NULL) return;

		PendingTasks.push(new Task{
			File, Optimize, Then,
			[](ResourceData & Data) -> std::future<bool> {
				std::future<bool> Task = std::async(std::launch::async, Load, std::ref(Data));
				return std::move(Task);
			}
			});
	}

	MeshParser::ResourceData::ResourceData(const FileStream * File, bool Optimize) :
		File(File), Optimize(Optimize), Meshes(), MeshTransforms(), bLoaded() {
	}

	MeshParser::ResourceData & MeshParser::ResourceData::operator=(ResourceData & Other) {
		File = Other.File;
		Optimize = Other.Optimize;
		Meshes.swap(Other.Meshes);
		MeshTransforms.swap(Other.MeshTransforms);
		bLoaded = Other.bLoaded;
		return *this;
	}

	MeshParser::Task::Task(const FileStream * File, bool Optimize, FinishTaskFunction FinishFunction, FutureTask Future) :
		Data(File, Optimize), FinishFunction(FinishFunction), Future(Future) {
	}

}