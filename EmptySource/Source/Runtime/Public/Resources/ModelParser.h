#pragma once

#include "Files/FileManager.h"
#include "Core/Transform.h"
#include "Resources/ModelResource.h"

#include <queue>
#include <future>

namespace ESource {

	class ModelParser {
	public:
		struct ParsingOptions {
			const FileStream * File;
			bool Optimize;
		};

		struct ModelDataInfo {
			TArray<MeshData> Meshes;
			ModelNode ParentNode;

			//* The model data has been succesfully loaded
			bool bSuccess;
			bool bAnimated;

			void Transfer(ModelDataInfo & Other);

			ModelDataInfo();
			
			ModelDataInfo(const ModelDataInfo & Other) = delete;
		};

		struct AnimationInfo {
			struct AnimationNode {
				NString Name;
				TArray<Vector3> Positions;
				TArray<Quaternion> Rotations;
				TArray<Vector3> Scalings;
			};
			double Duration;
			double TicksPerSecond;
			TArray<AnimationNode> Channels;
		};

	private:
		typedef std::function<void(ModelDataInfo &)> FinishTaskFunction;
		typedef std::function<std::future<bool>(ModelDataInfo &, const ParsingOptions &)> FutureTask;

		static bool _TaskRunning;

		struct Task {
			ParsingOptions Options;
			ModelDataInfo Info;
			FinishTaskFunction FinishFunction;
			FutureTask Future;

			Task(const Task& Other) = delete;
			Task(const ParsingOptions & Options, FinishTaskFunction FinishFunction, FutureTask Future);
		};

		static bool RecognizeFileExtensionAndLoad(ModelDataInfo & Data, const ParsingOptions & Options);

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

		static bool Load(ModelDataInfo & Info, const ParsingOptions & Options);

		static void LoadAsync(const ParsingOptions & Options, FinishTaskFunction OnComplete);

	};

}