#pragma once

#include "Utility/TextFormatting.h"
#include "Core/IIdentifier.h"
#include "Resources/ResourceManager.h"

namespace EmptySource {

	class ResourceHolder {
	protected:
		friend class ResourceManager;

		const WString Name;

		ResourceHolder(const WString& Name);

	public:
		virtual ~ResourceHolder() = default;

		virtual void Load() = 0;

		virtual void Unload() = 0;

		virtual void Reload() = 0;

		virtual inline EResourceType GetResourceType() const = 0;

		virtual inline EResourceLoadState GetLoadState() const = 0;

		inline WString GetFriendlyName() const { return Name; };

	};

}