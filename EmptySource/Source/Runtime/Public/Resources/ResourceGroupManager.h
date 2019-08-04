#pragma once

#include "Resources/ResourceHolder.h"
#include "Engine/CoreTypes.h"
#include "Engine/Text.h"

namespace EmptySource {

	class ResourceGroupManager {
	private:
		typedef TDictionary<String, TArray<ResourceHolder *>> GroupDictionary;

		GroupDictionary GroupDictionary;
	};

}