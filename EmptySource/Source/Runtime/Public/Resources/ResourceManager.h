#pragma once

#include "Core/IIdentifier.h"
#include "Files/FileManager.h"

namespace EmptySource {

	enum EResourceLoadState {
		LS_Loaded, LS_Loading, LS_Unloaded, LS_Unloading
	};

	enum EResourceType {
		RT_Texture, RT_Shader, RT_Material, RT_Mesh
	};

	class ResourceManager {
	public:
		virtual inline EResourceType GetResourceType() const = 0;

		virtual void GetResourcesFromFile(const WString& FilePath) = 0;

		static FileStream * GetResourcesFile(const WString & FilePath);

		static FileStream * CreateResourcesFile(const WString & FilePath);
	};

}