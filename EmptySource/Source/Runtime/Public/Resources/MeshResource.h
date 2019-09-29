#pragma once

#include "Resources/ResourceHolder.h"
#include "Rendering/Mesh.h"

namespace ESource {

	typedef std::shared_ptr<class RShader> RMeshPtr;

	class RMesh : public ResourceHolder {

		~RMesh();

		virtual bool IsValid() const override;

		virtual void Load() override;

		virtual void LoadAsync() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }

		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Mesh; }

		virtual inline size_t GetMemorySize() const override;

		static inline EResourceType GetType() { return EResourceType::RT_Mesh; };

	protected:
		friend class MeshManager;

		RMesh(const IName & Name, const WString & Origin);

	private:
		Mesh * MeshPointer;
	};

}