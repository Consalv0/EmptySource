#pragma once

#include "../include/Text.h"
#include "../include/IIdentifier.h"
#include "../include/ResourceManager.h"
#include "../include/Property.h"

namespace EmptySource {

	class ResourceHolder : public IIdentifier {
	protected:
		friend class ResourceManager;

		Property<ResourceLoadState> LoadState;
		ResourceManager * Manager;
		const WString Name;

		ResourceHolder(ResourceManager * Manager, const WString & Name);

	public:

		virtual void Load() = 0;

		virtual void Unload() = 0;

		virtual void Reload() = 0;

		const ResourceLoadState & GetLoadState() const { return LoadState.Get(); };

		WString GetFriendlyName() const { return Name; };

		ResourceManager * GetResourceManager() const { return Manager; };
	};

}