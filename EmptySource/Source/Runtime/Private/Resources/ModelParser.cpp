
#include "CoreMinimal.h"
#include "Rendering/Mesh.h"
#include "Rendering/Animation.h"
#include "Resources/ModelParser.h"
#include "Resources/OBJLoader.h"
#include "Resources/FBXLoader.h"

#include <future>
#include <thread>

#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "assimp/Importer.hpp"
#include "assimp/DefaultLogger.hpp"
#include "assimp/LogStream.hpp"

namespace ESource {

	bool ModelParser::_TaskRunning;
	std::queue<ModelParser::Task*> ModelParser::PendingTasks = std::queue<Task*>();
	std::future<bool> ModelParser::CurrentFuture;
	std::mutex QueueLock;

	static Matrix4x4 aiMatrix4x4ToMatrix4x4(const aiMatrix4x4 &From) {
		return Matrix4x4(
			From.a1, From.b1, From.c1, From.d1,
			From.a2, From.b2, From.c2, From.d2,
			From.a3, From.b3, From.c3, From.d3,
			From.a4, From.b4, From.c4, From.d4
		);
	}

	void FillNodeInfo(const aiScene * AssimpScene, aiNode* Node, ModelParser::ModelDataInfo & Info, ModelNode * InfoNode, int Level = 0) {
#ifdef ES_DEBUG
		NString IndentText;
		for (int i = 0; i < Level; i++)
			IndentText += "-";
		LOG_CORE_DEBUG("{0}Node name: {1}", IndentText, NString(Node->mName.data));
#endif // ES_DEBUG

		if (InfoNode == NULL) InfoNode = &Info.ParentNode;
		else InfoNode = InfoNode->AddChild(Node->mName.C_Str());
		InfoNode->LocalTransform = aiMatrix4x4ToMatrix4x4(Node->mTransformation);
		InfoNode->bHasMesh = Node->mNumMeshes > 0;
		InfoNode->MeshKey = Info.Meshes.size();

		uint32_t VertexCount = 0;
		uint32_t IndexCount = 0;

		if (Node->mMeshes > 0) {
			Info.Meshes.push_back(MeshData());
			Info.Meshes.back().Name = Node->mName.C_Str();
		}
		for (uint32_t m = 0; m < Node->mNumMeshes; m++) {
			aiMesh* Mesh = AssimpScene->mMeshes[Node->mMeshes[m]];

			MeshData & Data = Info.Meshes.back();

			Data.SubdivisionsMap[m].MaterialIndex = m;
			if (AssimpScene->HasMaterials())
				Data.MaterialsMap.insert_or_assign(m, AssimpScene->mMaterials[Mesh->mMaterialIndex]->GetName().C_Str());
			else
				Data.MaterialsMap.insert_or_assign(m, "BaseMaterial");
			Data.SubdivisionsMap[m].BaseVertex = VertexCount;
			Data.SubdivisionsMap[m].BaseIndex = IndexCount;
			Data.SubdivisionsMap[m].IndexCount = Mesh->mNumFaces * 3;

			VertexCount += Mesh->mNumVertices;
			IndexCount += Mesh->mNumFaces * 3;

			Data.hasNormals = Mesh->HasNormals();
			Data.hasTangents = Mesh->HasTangentsAndBitangents();
			Data.hasVertexColor = Mesh->HasVertexColors(0);
			Data.UVChannels = Mesh->GetNumUVChannels();
			Data.hasBones = Mesh->HasBones();
			Data.hasBoundingBox = true;

			// Vertices
			for (size_t i = 0; i < Mesh->mNumVertices; i++) {
				StaticVertex Vertex;
				Vertex.Position = { Mesh->mVertices[i].x, Mesh->mVertices[i].y, Mesh->mVertices[i].z };
				Data.Bounding.Add(Vertex.Position);

				if (Data.hasNormals)
					Vertex.Normal = { Mesh->mNormals[i].x, Mesh->mNormals[i].y, Mesh->mNormals[i].z };

				if (Data.hasTangents)
					Vertex.Tangent = { Mesh->mTangents[i].x, Mesh->mTangents[i].y, Mesh->mTangents[i].z };

				if (Mesh->HasTextureCoords(0))
					Vertex.UV0 = { Mesh->mTextureCoords[0][i].x, Mesh->mTextureCoords[0][i].y };
				if (Mesh->HasTextureCoords(1))
					Vertex.UV1 = { Mesh->mTextureCoords[1][i].x, Mesh->mTextureCoords[1][i].y };

				if (Mesh->HasVertexColors(0))
					Vertex.Color = { Mesh->mColors[0][i].r, Mesh->mColors[0][i].g, Mesh->mColors[0][i].b, Mesh->mColors[0][i].a };
				else
					Vertex.Color = 1.0F;

				Data.StaticVertices.push_back(Vertex);
				if (Info.bHasAnimations)
					Data.SkinVertices.push_back(Vertex);
			}

			// Indices
			for (size_t i = 0; i < Mesh->mNumFaces; i++) {
				Data.Faces.push_back({ (int)Mesh->mFaces[i].mIndices[0], (int)Mesh->mFaces[i].mIndices[1], (int)Mesh->mFaces[i].mIndices[2] });
			}

		}

		for (uint32_t i = 0; i < Node->mNumChildren; i++) {
			aiNode* Child = Node->mChildren[i];
			FillNodeInfo(AssimpScene, Child, Info, InfoNode, Level + 1);
		}
	}

	static const uint32_t MeshImportFlags =
		aiProcess_CalcTangentSpace |        // Create binormals/tangents just in case
		aiProcess_Triangulate |             // Make sure we're triangles
		aiProcess_SortByPType |             // Split meshes by primitive type
		aiProcess_GenNormals |              // Make sure we have legit normals
		aiProcess_GenUVCoords |             // Convert UVs if required
		aiProcess_GlobalScale |
		aiProcess_OptimizeMeshes |          // Batch draws where possible
		aiProcess_ValidateDataStructure;    // Validation
	
	struct AssimpLogStream : public Assimp::LogStream {
		static void Initialize() {
			if (Assimp::DefaultLogger::isNullLogger()) {
				Assimp::DefaultLogger::create("", Assimp::Logger::VERBOSE);
				Assimp::DefaultLogger::get()->attachStream(new AssimpLogStream, Assimp::Logger::Err);
			}
		}

		virtual void write(const char* message) override {
			LOG_CORE_ERROR("Assimp: {0}", message);
		}
	};

	bool FindNodeLevel(const NString & Name, const ModelNode * Node, int & Level) {
		if (Node->Name == Name)
			return true;
		for (auto & Child : Node->Children) {
			bool Find = FindNodeLevel(Name, Child, ++Level);
			if (Find) {
				return true;
			}
		}
		Level--;
		return false;
	}
	
	void GetAnimationLevels(ModelParser::ModelDataInfo & Info) {
		for (AnimationTrack & Track : Info.Animations) {
			for (AnimationTrackNode & NodeA : Track.AnimationNodes) {
				FindNodeLevel(NodeA.Name.GetNarrowDisplayName(), &Info.ParentNode, NodeA.NodeLevel);
			}
		}
	}

	bool LoadWithAssimp(ModelParser::ModelDataInfo & Info, const ModelParser::ParsingOptions & Options) {
		auto Importer = std::make_unique<Assimp::Importer>();
		const aiScene* AssimpScene = Importer->ReadFile(Text::WideToNarrow(Options.File->GetPath()), MeshImportFlags);
		if (Info.bSuccess = !AssimpScene) return false;
		
		Info.bHasAnimations = AssimpScene->HasAnimations();
		
		FillNodeInfo(AssimpScene, AssimpScene->mRootNode, Info, NULL);
		
		if (Info.bHasAnimations) {
			for (unsigned int a = 0; a < AssimpScene->mNumAnimations; ++a) {
				Info.Animations.push_back(AnimationTrack());
				AnimationTrack & Animation = Info.Animations.back();
				aiAnimation * Anim = AssimpScene->mAnimations[a];
				Animation.Name = Anim->mName.C_Str();
				Animation.Duration = Anim->mDuration;
				Animation.TicksPerSecond = Anim->mTicksPerSecond;
		
				for (unsigned int n = 0; n < Anim->mNumChannels; ++n) {
					aiNodeAnim * AnimNode = Anim->mChannels[n];
					Animation.AnimationNodes.push_back(AnimationTrackNode(Animation, Text::NarrowToWide(AnimNode->mNodeName.C_Str())));
					AnimationTrackNode & AnimationNode = Animation.AnimationNodes.back();
					AnimationNode.Positions.reserve(AnimNode->mNumPositionKeys);
					for (unsigned int i = 0; i < AnimNode->mNumPositionKeys; i++) {
						aiVectorKey & Key = AnimNode->mPositionKeys[i];
						AnimationNode.Positions.emplace_back(AnimationNode, Key.mTime, Key.mValue.x, Key.mValue.y, Key.mValue.z);
					}
					AnimationNode.Rotations.reserve(AnimNode->mNumRotationKeys);
					for (unsigned int i = 0; i < AnimNode->mNumRotationKeys; i++) {
						aiQuatKey & Key = AnimNode->mRotationKeys[i];
						AnimationNode.Rotations.emplace_back(AnimationNode, Key.mTime, Key.mValue.w, Key.mValue.x, Key.mValue.y, Key.mValue.z);
					}
					AnimationNode.Scalings.reserve(AnimNode->mNumScalingKeys);
					for (unsigned int i = 0; i < AnimNode->mNumScalingKeys; i++) {
						aiVectorKey & Key = AnimNode->mScalingKeys[i];
						AnimationNode.Scalings.emplace_back(AnimationNode, Key.mTime, Key.mValue.x, Key.mValue.y, Key.mValue.z);
					}
		
				}
			}
		
			GetAnimationLevels(Info);
		
		}

		return true;
	}

	bool ModelParser::RecognizeFileExtensionAndLoad(ModelDataInfo & Info, const ParsingOptions & Options) {
		const WString Extension = Options.File->GetExtension();
		if (Text::CompareIgnoreCase(Extension, WString(L"FBX"))) {
			// return FBXLoader::LoadModel(Info, Options);
		}
		if (Text::CompareIgnoreCase(Extension, WString(L"OBJ"))) {
			return OBJLoader::LoadModel(Info, Options);
		}

		return LoadWithAssimp(Info, Options);;
	}

	bool ModelParser::Initialize() {
		if (std::thread::hardware_concurrency() <= 1) {
			LOG_CORE_WARN(L"The aviable cores ({:d}) are insuficient for asyncronus loaders", std::thread::hardware_concurrency());
			return false;
		}

		AssimpLogStream::Initialize();
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
		: Meshes(), ParentNode("ParentNode"), bSuccess(false) {
	}

	void ModelParser::ModelDataInfo::Transfer(ModelDataInfo & Other) {
		Meshes.clear();
		ParentNode = Other.ParentNode;
		Meshes.swap(Other.Meshes);
		bSuccess = Other.bSuccess;
		bHasAnimations = Other.bHasAnimations;
	}

	ModelParser::Task::Task(const ParsingOptions & Options, FinishTaskFunction FinishFunction, FutureTask Future) :
		Info(), Options(Options), FinishFunction(FinishFunction), Future(Future) {
	}

}