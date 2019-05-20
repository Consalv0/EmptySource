#pragma once

#include "../include/Core.h"
#include "../include/IIdentifier.h"
#include "../include/FileManager.h"

struct FileStream;
class ResourceManager;

enum ResourceType {
	RT_Texture, RT_ShaderPass, RT_ShaderProgram, RT_Material, RT_Mesh
};

struct BaseResource : public IIdentifier {
protected:
	friend class ResourceManager;
	
	BaseResource(const WString & FilePath);
	virtual ~BaseResource() { }
	const WString FilePath;
	bool isDone;

public:
	bool IsDone();
	const FileStream * GetFile();
};

template <typename T>
struct Resource : public BaseResource {
protected:
	friend class ResourceManager;
	Resource(const WString & FilePath) : BaseResource(FilePath), Data(NULL) {}
	Resource(const WString & FilePath, T * Data) : BaseResource(FilePath), Data(Data) { isDone = true; }

	T * Data;
public:

	T * GetData() {
		if (!isDone) return NULL;
		return Data;
	}
};

class ResourceManager {
public:
	template<typename T>
	static inline Resource<T> * Load(const WString & File);

private:
	template<typename T>
	static bool ResourceToList(const WString & File, T & Data);

	static TDictionary<size_t, BaseResource*> Resources;
};

#include "../include/ResourceShaderStage.h"
#include "../include/ResourceShaderProgram.h"