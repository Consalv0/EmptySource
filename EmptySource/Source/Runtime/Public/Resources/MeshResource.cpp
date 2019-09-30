
#include "CoreMinimal.h"
#include "Resources/ModelManager.h"
#include "Resources/MeshResource.h"

namespace ESource {

	bool RMesh::IsValid() const {
		return LoadState == LS_Loaded && VAOSubdivisions.size() > 0;
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
			else if (VertexData.Vertices.size() > 0){
				SetUpBuffers();
			}
		}
		LoadState = VAOSubdivisions.size() > 0 ? LS_Loaded : LS_Unloaded;
	}

	void RMesh::LoadAsync() {
		ES_CORE_ASSERT(true, "No implemented");
	}

	void RMesh::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		VAOSubdivisions.clear();
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

	VertexArrayPtr RMesh::GetSubdivisionVertexArray(int Index) const {
		auto Subdivision = VertexData.Subdivisions.find(Index);
		if (Subdivision == VertexData.Subdivisions.end()) return NULL;
		if (Index >= VAOSubdivisions.size() || VAOSubdivisions[Index] == NULL) return NULL;

		return VAOSubdivisions[Index];
	}

	bool RMesh::SetUpBuffers() {
		if (VertexData.Vertices.size() <= 0 || VertexData.Faces.size() <= 0) return false;

		static BufferLayout DafultLayout = {
			{ EShaderDataType::Float3, "_iVertexPosition" },
			{ EShaderDataType::Float3, "_iVertexNormal", true },
			{ EShaderDataType::Float3, "_iVertexTangent", true },
			{ EShaderDataType::Float2, "_iVertexUV0" },
			{ EShaderDataType::Float2, "_iVertexUV1" },
			{ EShaderDataType::Float4, "_iVertexColor" }
		};

		// Give our vertices to VAO
		VertexBufferPtr VertexBufferPointer = NULL;
		VertexBufferPointer = VertexBuffer::Create((float *)&VertexData.Vertices[0], (unsigned int)(VertexData.Vertices.size() * sizeof(MeshVertex)), UM_Static);
		VertexBufferPointer->SetLayout(DafultLayout);

		if (VertexData.Subdivisions.size() <= 0) {
			VertexData.Materials.emplace(0, "default");
			VertexData.Subdivisions.emplace(0, VertexData.Faces);
		}

		for (int ElementBufferCount = 0; ElementBufferCount < VertexData.Subdivisions.size(); ElementBufferCount++) {
			VertexArrayPtr VertexArrayPointer = NULL;
			VertexArrayPointer = VertexArray::Create();
			IndexBufferPtr IndexBufferPointer = NULL;
			IndexBufferPointer = IndexBuffer::Create(
				(unsigned int *)&VertexData.Subdivisions[ElementBufferCount][0],
				(unsigned int)VertexData.Subdivisions[ElementBufferCount].size() * 3, UM_Static
			);
			VertexArrayPointer->AddVertexBuffer(VertexBufferPointer);
			VertexArrayPointer->AddIndexBuffer(IndexBufferPointer);
			VertexArrayPointer->Unbind();

			VAOSubdivisions.push_back(VertexArrayPointer);
		}

		return true;
	}

	void RMesh::ClearBuffers() {
		VAOSubdivisions.clear();
	}

	void RMesh::Clear() {
		VertexData.Clear();
		ClearBuffers();
	}

	RMesh::RMesh(const IName & Name, const WString & Origin, const IName & InModelName, MeshData & InVertexData)
		: ResourceHolder(Name, Origin), ModelName(InModelName), VAOSubdivisions() {
		VertexData.Transfer(InVertexData);
	}

	RMesh::RMesh(const IName & Name, const WString & Origin, const IName & InModelName)
		: ResourceHolder(Name, Origin), ModelName(InModelName), VAOSubdivisions(), VertexData() {
	}

	RMesh::~RMesh() {
		Unload();
	}

}