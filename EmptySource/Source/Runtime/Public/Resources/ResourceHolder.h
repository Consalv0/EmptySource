#pragma once

#include "Utility/TextFormatting.h"
#include "Core/IIdentifier.h"
#include "Resources/ResourceManager.h"

namespace EmptySource {

	class Resource {
	protected:
		friend class ResourceManager;

		const WString Name;

		const size_t UID;

		Resource(const WString& Name);

	public:
		virtual ~Resource() = default;

		virtual void Load() = 0;

		virtual void Unload() = 0;

		virtual void Reload() = 0;

		virtual inline EResourceType GetResourceType() const = 0;

		virtual inline EResourceLoadState GetLoadState() const = 0;

		virtual size_t GetMemorySize() const = 0;

		inline WString GetName() const { return Name; };

	};

	typedef std::shared_ptr<Resource> ResourcePtr;

}