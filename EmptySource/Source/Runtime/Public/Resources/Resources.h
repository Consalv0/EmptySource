#pragma once

#include "CoreMinimal.h"
#include "Engine/IIdentifier.h"
#include "Files/FileManager.h"

namespace EmptySource {

	struct FileStream;
	class ResourceManager;

	struct BaseResource {
	protected:
		friend class OldResourceManager;

		BaseResource(const WString & FilePath, const size_t & GUID);
		virtual ~BaseResource() { }
		const WString Name;
		const size_t GUID;
		bool isDone;

	public:
		bool IsDone() const;
		size_t GetIdentifier() const;
		const FileStream * GetFile() const;
	};

	template <typename T>
	struct Resource : public BaseResource {
	protected:
		friend class OldResourceManager;
		Resource(const WString & FilePath, const size_t & GUID) : BaseResource(FilePath, GUID), Data(NULL) {}
		Resource(const WString & FilePath, const size_t & GUID, T * Data) : BaseResource(FilePath, GUID), Data(Data) { isDone = true; }

		T * Data;
	public:

		T * GetData() {
			if (!isDone) return NULL;
			return Data;
		}
	};

	class OldResourceManager {
	public:
		//* Get the resource with the given name, returns NULL if no resource
		template<typename T>
		static inline Resource<T> * Load(const WString & File);
		//* Get the resource with the given GUID. 
		//  Use this if you are shure the file is in Resources.yaml, returns NULL if no resource
		template<typename T>
		static inline Resource<T> * Load(const size_t & GUID);

		template<typename T>
		static inline Resource<T> * Load(const WString& Name, T * Data) {
			size_t GUID = WStringToHash(Name);
			auto ResourceFind = Resources.find(GUID);
			if (ResourceFind != Resources.end()) {
				return dynamic_cast<Resource<T>*>(ResourceFind->second);
			}

			Resource<T> * ResourceAdded = new Resource<T>(Name, GUID, Data);
			Resources.insert(std::pair<const size_t, BaseResource*>(ResourceAdded->GetIdentifier(), ResourceAdded));
			return ResourceAdded;
		}

		template<typename T>
		static inline Resource<T> * Get(const WString& Name) {
			size_t GUID = WStringToHash(Name);
			auto ResourceFind = Resources.find(GUID);
			if (ResourceFind != Resources.end()) {
				return dynamic_cast<Resource<T>*>(ResourceFind->second);
			}

			return NULL;
		}

	private:
		template<typename T>
		static bool GetResourceData(const WString & File, T & OutData);
		template<typename T>
		static bool GetResourceData(const size_t & GUID, T & OutData);

		static FileStream * GetResourcesFile();

		static TDictionary<size_t, BaseResource*> Resources;
	};

}

#include "Resources/ResourceShaderStage.h"
#include "Resources/ResourceShaderProgram.h"