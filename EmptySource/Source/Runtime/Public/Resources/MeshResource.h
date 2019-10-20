#pragma once

#include "Resources/ResourceHolder.h"
#include "Resources/ModelResource.h"
#include "Rendering/Mesh.h"

namespace ESource {

	typedef std::shared_ptr<class RMesh> RMeshPtr;

	class RMesh : public ResourceHolder {
	public:
		~RMesh();

		virtual bool IsValid() const override;

		virtual void Load() override;

		virtual void LoadAsync() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }

		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Mesh; }

		static inline EResourceType GetType() { return EResourceType::RT_Mesh; };

		virtual inline size_t GetMemorySize() const override;

		const IName & GetModelName() const;

		MeshData & GetVertexData();

		//* Get VertexArray in Mesh
		VertexArrayPtr GetVertexArray() const { return VertexArrayPointer; };

	protected:
		friend class ModelManager;

		RMesh(const IName & Name, const WString & Origin, const IName & ModelName, MeshData & VertexData);

		RMesh(const IName & Name, const WString & Origin, const IName & ModelName);

		//* Clear the mesh entirely
		void Clear();

		//* Clear the GL's objects
		void ClearBuffers();

		//* Give Vertices to OpenGL **This must be done once per render**
		bool SetUpBuffers();

	private:
		VertexArrayPtr VertexArrayPointer;

		IName ModelName;
		
		MeshData VertexData;
	};

}