#pragma once

#include "Utility/TextFormatting.h"
#include "Core/Name.h"
#include "Resources/ResourceManager.h"

namespace ESource {

	class ResourceHolder {
	public:
		virtual ~ResourceHolder() = default;

		virtual bool IsValid() const = 0;

		virtual void Load() = 0;

		virtual void LoadAsync() = 0;

		virtual void Unload() = 0;

		virtual void Reload() = 0;

		virtual inline EResourceType GetResourceType() const = 0;

		virtual inline EResourceLoadState GetLoadState() const = 0;

		virtual size_t GetMemorySize() const = 0;

		inline const IName & GetName() const { return Name; };

	protected:
		friend class ResourceManager;

		IName Name;

		const WString Origin;

		EResourceLoadState LoadState;

		ResourceHolder(const IName& Name, const WString& Origin);

	};

	typedef std::shared_ptr<ResourceHolder> ResourcePtr;

}