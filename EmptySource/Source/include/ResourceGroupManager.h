#pragma once

#include "../include/ResourceHolder.h"
#include "../include/CoreTypes.h"
#include "../include/Text.h"

namespace EmptySource {

	class ResourceGroupManager {
	private:
		typedef TDictionary<String, TArray<ResourceHolder *>> GroupDictionary;

		GroupDictionary GroupDictionary;
	};

}