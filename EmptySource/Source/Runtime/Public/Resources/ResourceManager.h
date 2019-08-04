#pragma once

#include "Engine/IIdentifier.h"
#include "Files/FileManager.h"

namespace EmptySource {

	enum ResourceLoadState {
		LS_Loaded, LS_Loading, LS_Unloaded, LS_Unloading
	};

	enum ResourceType {
		RT_Texture, RT_ShaderStage, RT_ShaderProgram, RT_Material, RT_Mesh
	};

	class ResourceManager {
	protected:
		const unsigned int LoadOrder;

		const ResourceType Type;

		ResourceManager(unsigned int LoadOrder, ResourceType Type) : LoadOrder(LoadOrder), Type(Type) {};

	public:
		//* Lower value indicate it will be loaded firts
		const unsigned int & GetLoadOrder() const { return LoadOrder; };

		const ResourceType & GetResourceType() const { return Type; };

		virtual class ResourceHolder * GetResourceByUniqueName(const WString& Name) const { return NULL; };

		virtual class ResourceHolder * GetResourceByUniqueID(const size_t & UID) const { return NULL; };

		static FileStream * GetResourcesFile(const WString & FilePath);

		static FileStream * CreateResourceFile(const WString & FilePath);
	};

}