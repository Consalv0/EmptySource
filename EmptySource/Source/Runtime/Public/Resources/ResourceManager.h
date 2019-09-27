#pragma once

#include "Files/FileManager.h"

namespace ESource {

	enum EResourceLoadState {
		LS_Loaded, LS_Loading, LS_Unloaded, LS_Unloading
	};

	enum EResourceType {
		RT_Texture, RT_Shader, RT_Material, RT_Mesh, RT_Audio
	};

	class ResourceManager {
	public:
		virtual inline EResourceType GetResourceType() const = 0;

		virtual void LoadResourcesFromFile(const WString& FilePath) = 0;

		static FileStream * GetResourcesFile(const WString & FilePath);

		static FileStream * CreateResourcesFile(const WString & FilePath);
	};

}