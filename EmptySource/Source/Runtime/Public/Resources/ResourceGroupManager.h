#pragma once

#include "Resources/ResourceHolder.h"
#include "Core/CoreTypes.h"
#include "Core/Text.h"

namespace EmptySource {

	class ResourceGroupManager {
	private:
		typedef TDictionary<String, TArray<ResourceHolder *>> GroupDictionary;

		GroupDictionary GroupDictionary;
	};

}