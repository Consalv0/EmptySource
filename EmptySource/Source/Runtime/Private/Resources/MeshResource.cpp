
#include "CoreMinimal.h"
#include "Resources/ModelManager.h"
#include "Resources/MeshResource.h"

namespace ESource {

	bool RMesh::IsValid() const {
		return LoadState == LS_Loaded && VertexArrayPointer != NULL;
	}

	void RMesh::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			LOG_CORE_DEBUG(L"Loading Mesh {}...", Name.GetDisplayName().c_str());
			if (!Origin.empty()) {
				RModelPtr Model = ModelManager::GetInstance().GetModel(ModelName);
				if (Model != NULL) {
					SetUpBuffers();
				}
			}
			else if (VertexData.StaticVertices.size() > 0){
				SetUpBuffers();
			}
		}
		LoadState = VertexArrayPointer != NULL ? LS_Loaded : LS_Unloaded;
	}

	void RMesh::LoadAsync() {
		ES_CORE_ASSERT(true, "No implemented");
	}

	void RMesh::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		VertexArrayPointer = NULL;
		VertexData.Clear();
		LoadState = LS_Unloaded;
	}

	void RMesh::Reload() {
		Unload();
		Load();
	}

	inline size_t RMesh::GetMemorySize() const {
		return size_t();
	}

	const IName & RMesh::GetModelName() const {
		return ModelName;
	}

	MeshData & RMesh::GetVertexData() {
		return VertexData;
	}

	bool RMesh::SetUpBuffers() {
		if (VertexData.StaticVertices.size() <= 0 || VertexData.Faces.size() <= 0) return false;

		static BufferLayout DefaultLayout = {
			{ EShaderDataType::Float3, "_iVertexPosition" },
			{ EShaderDataType::Float3, "_iVertexNormal", true },
			{ EShaderDataType::Float3, "_iVertexTangent", true },
			{ EShaderDataType::Float2, "_iVertexUV0" },
			{ EShaderDataType::Float2, "_iVertexUV1" },
			{ EShaderDataType::Float4, "_iVertexColor" }
		};

		// Give our vertices to VAO
		VertexBufferPtr VertexBufferPointer = NULL;
		VertexBufferPointer = VertexBuffer::Create((float *)&VertexData.StaticVertices[0], (uint32_t)(VertexData.StaticVertices.size() * sizeof(StaticVertex)), UM_Static);
		VertexBufferPointer->SetLayout(DefaultLayout);

		VertexArrayPointer = VertexArray::Create();
		IndexBufferPtr IndexBufferPointer = IndexBuffer::Create(
			(uint32_t *)&VertexData.Faces[0],
			(uint32_t)VertexData.Faces.size() * 3, UM_Static
		);
		VertexArrayPointer->AddVertexBuffer(VertexBufferPointer);
		VertexArrayPointer->AddIndexBuffer(IndexBufferPointer);
		VertexArrayPointer->Unbind();

		return true;
	}

	void RMesh::ClearBuffers() {
		VertexArrayPointer = NULL;
	}

	void RMesh::Clear() {
		VertexData.Clear();
		ClearBuffers();
	}

	RMesh::RMesh(const IName & Name, const WString & Origin, const IName & InModelName, MeshData & InVertexData)
		: ResourceHolder(Name, Origin), ModelName(InModelName), VertexArrayPointer(NULL) {
		VertexData.Transfer(InVertexData);
	}

	RMesh::RMesh(const IName & Name, const WString & Origin, const IName & InModelName)
		: ResourceHolder(Name, Origin), ModelName(InModelName), VertexArrayPointer(NULL), VertexData() {
	}

	RMesh::~RMesh() {
		Unload();
	}

}