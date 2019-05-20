#pragma once

#include "../include/Core.h"
#include "../include/IIdentifier.h"
#include "../include/FileManager.h"

struct FileStream;
class ResourceManager;

enum ResourceType {
	RT_Texture, RT_ShaderPass, RT_ShaderProgram, RT_Material, RT_Mesh
};

struct BaseResource {
protected:
	friend class ResourceManager;
	
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
	friend class ResourceManager;
	Resource(const WString & FilePath, const size_t & GUID) : BaseResource(FilePath, GUID), Data(NULL) {}
	Resource(const WString & FilePath, const size_t & GUID, T * Data) : BaseResource(FilePath, GUID), Data(Data) { isDone = true; }

	T * Data;
public:

	T * GetData() {
		if (!isDone) return NULL;
		return Data;
	}
};

class ResourceManager {
public:
	//* Get the resource with the given name, returns NULL if no resource
	template<typename T>
	static inline Resource<T> * Load(const WString & File);
	//* Get the resource with the given name use this if you are shure the file is in Resources.yaml, returns NULL if no resource
	template<typename T>
	static inline Resource<T> * Load(const size_t & GUID);

private:
	template<typename T>
	static bool GetResourceData(const WString & File, T & OutData);
	template<typename T>
	static bool GetResourceData(const size_t & GUID, T & OutData);

	static FileStream * GetResourcesFile();

	static TDictionary<size_t, BaseResource*> Resources;
};

#include "../include/ResourceShaderStage.h"
#include "../include/ResourceShaderProgram.h"