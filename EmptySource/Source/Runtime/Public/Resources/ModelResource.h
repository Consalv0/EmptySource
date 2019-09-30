#pragma once

#include "Resources/MaterialManager.h"
#include "Resources/ResourceHolder.h"
#include "Resources/MeshResource.h"

namespace ESource {

	typedef std::shared_ptr<class RModel> RModelPtr;

	class RModel : public ResourceHolder {
	public:

		bool bOptimizeOnLoad;

		~RModel();

		virtual bool IsValid() const override;

		virtual void Load() override;

		virtual void LoadAsync() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }

		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Model; }

		virtual inline size_t GetMemorySize() const override;

		static inline EResourceType GetType() { return EResourceType::RT_Model; };

	protected:
		friend class ModelManager;

		RModel(const IName & Name, const WString & Origin, bool bOptimize = false);

	private:
		TDictionary<size_t, RMeshPtr> Meshes;

		TDictionary<NString, Material> DefaultMaterials;

	};

}