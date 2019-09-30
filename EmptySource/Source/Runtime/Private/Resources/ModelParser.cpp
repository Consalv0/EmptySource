
#include "CoreMinimal.h"
#include "Rendering/Mesh.h"
#include "Resources/ModelParser.h"
#include "Resources/OBJLoader.h"
#include "Resources/FBXLoader.h"

#include <future>
#include <thread>

namespace ESource {

	bool ModelParser::_TaskRunning;
	std::queue<ModelParser::Task*> ModelParser::PendingTasks = std::queue<Task*>();
	std::future<bool> ModelParser::CurrentFuture;
	std::mutex QueueLock;

	bool ModelParser::RecognizeFileExtensionAndLoad(ModelDataInfo & Info, const ParsingOptions & Options) {
		const WString Extension = Options.File->GetExtension();
		if (Text::CompareIgnoreCase(Extension, WString(L"FBX"))) {
			return FBXLoader::LoadModel(Info, Options);
		}
		if (Text::CompareIgnoreCase(Extension, WString(L"OBJ"))) {
			return OBJLoader::LoadModel(Info, Options);
		}

		return false;
	}

	bool ModelParser::Initialize() {
		if (std::thread::hardware_concurrency() <= 1) {
			LOG_CORE_WARN(L"The aviable cores ({:d}) are insuficient for asyncronus loaders", std::thread::hardware_concurrency());
			return false;
		}
		if (!FBXLoader::InitializeSdkManager()) {
			return false;
		}

		return true;
	}

	void ModelParser::UpdateStatus() {
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

	void ModelParser::FinishAsyncTasks() {
		do {
			FinishCurrentAsyncTask();
			UpdateStatus();
		} while (!PendingTasks.empty());
	}

	void ModelParser::FinishCurrentAsyncTask() {
		if (!PendingTasks.empty() && CurrentFuture.valid()) {
			CurrentFuture.get();
			PendingTasks.front()->FinishFunction(PendingTasks.front()->Info);
			delete PendingTasks.front();
			PendingTasks.pop();
		}
	}

	size_t ModelParser::GetAsyncTaskCount() {
		return PendingTasks.size();
	}

	void ModelParser::Exit() {
		if (CurrentFuture.valid())
			CurrentFuture.get();
	}

	bool ModelParser::Load(ModelDataInfo & Info, const ParsingOptions & Options) {
		if (Options.File == NULL) return false;

		if (_TaskRunning) {
			FinishCurrentAsyncTask();
		}

		_TaskRunning = true;
		LOG_CORE_INFO(L"Reading File Model '{}'", Options.File->GetShortPath());
		RecognizeFileExtensionAndLoad(Info, Options);
		_TaskRunning = false;
		return Info.bSuccess;
	}

	void ModelParser::LoadAsync(const ParsingOptions & Options, FinishTaskFunction Then) {
		if (Options.File == NULL) return;

		PendingTasks.push(
			new Task { Options, Then, [](ModelDataInfo & Data, const ParsingOptions & Options) -> std::future<bool> {
				std::future<bool> Task = std::async(std::launch::async, Load, std::ref(Data), std::ref(Options));
				return std::move(Task);
				}
			}
		);
	}

	ModelParser::ModelDataInfo::ModelDataInfo() 
		: Meshes(), MeshTransforms(), bSuccess(false) {
	}

	void ModelParser::ModelDataInfo::Transfer(ModelDataInfo & Other) {
		Meshes.clear();
		MeshTransforms.clear();
		Meshes.swap(Other.Meshes);
		MeshTransforms.swap(Other.MeshTransforms);
		bSuccess = Other.bSuccess;
	}

	ModelParser::Task::Task(const ParsingOptions & Options, FinishTaskFunction FinishFunction, FutureTask Future) :
		Info(), Options(Options), FinishFunction(FinishFunction), Future(Future) {
	}

}