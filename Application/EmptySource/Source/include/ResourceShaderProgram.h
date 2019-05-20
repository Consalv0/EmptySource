#pragma once

#ifdef RESOURCES_ADD_SHADERPROGRAM
#define RESOURCES_ADD_SHADERSTAGE
#include "../include/Resources.h"
#include "../include/ShaderProgram.h"

struct ShaderProgramData {
	size_t GUID;
	WString Name;
	WString VertexShader;
	WString FragmentShader;
	WString ComputeShader;
	WString GeometryShader;
};

template<>
bool ResourceManager::GetResourceData<ShaderProgramData>(const WString & File, ShaderProgramData & ResourceData);

template<>
inline Resource<ShaderProgram> * ResourceManager::Load(const WString & Name) {
	ShaderProgramData LoadData;
	if (!GetResourceData<ShaderProgramData>(Name, LoadData))
		return NULL;

	auto ResourceFind = Resources.find(LoadData.GUID);
	if (ResourceFind != Resources.end()) {
		return dynamic_cast<Resource<ShaderProgram>*>(ResourceFind->second);
	}

	ShaderProgram * Program = new ShaderProgram(LoadData.Name);
	if (LoadData.VertexShader.size() > 0) {
		Resource<ShaderStage> * Stage = ResourceManager::Load<ShaderStage>(LoadData.VertexShader);
		if (Stage && Stage->GetData()) {
			Program->AppendStage(Stage->GetData());
		}
	}
	if (LoadData.FragmentShader.size() > 0) {
		Resource<ShaderStage> * Stage = ResourceManager::Load<ShaderStage>(LoadData.FragmentShader);
		if (Stage && Stage->GetData()) {
			Program->AppendStage(Stage->GetData());
		}
	}
	if (LoadData.ComputeShader.size() > 0) {
		Resource<ShaderStage> * Stage = ResourceManager::Load<ShaderStage>(LoadData.ComputeShader);
		if (Stage && Stage->GetData()) {
			Program->AppendStage(Stage->GetData());
		}
	}
	if (LoadData.GeometryShader.size() > 0) {
		Resource<ShaderStage> * Stage = ResourceManager::Load<ShaderStage>(LoadData.GeometryShader);
		if (Stage && Stage->GetData()) {
			Program->AppendStage(Stage->GetData());
		}
	}
	Program->Compile();

	Resource<ShaderProgram> * ResourceAdded = new Resource<ShaderProgram>(Name, LoadData.GUID, Program);
	Resources.insert(std::pair<const size_t, BaseResource*>(ResourceAdded->GetIdentifier(), ResourceAdded));
	return ResourceAdded;
}

template<>
bool ResourceManager::GetResourceData<ShaderProgramData>(const size_t & GUID, ShaderProgramData & ResourceData);

template<>
inline Resource<ShaderProgram> * ResourceManager::Load(const size_t & GUID) {
	ShaderProgramData LoadData;
	if (!GetResourceData<ShaderProgramData>(GUID, LoadData))
		return NULL;

	auto ResourceFind = Resources.find(LoadData.GUID);
	if (ResourceFind != Resources.end()) {
		return dynamic_cast<Resource<ShaderProgram>*>(ResourceFind->second);
	}

	ShaderProgram * Program = new ShaderProgram(LoadData.Name);
	if (LoadData.VertexShader.size() > 0) {
		Resource<ShaderStage> * Stage = ResourceManager::Load<ShaderStage>(WStringToHash(LoadData.VertexShader));
		if (Stage && Stage->GetData()) {
			Program->AppendStage(Stage->GetData());
		}
	}
	if (LoadData.FragmentShader.size() > 0) {
		Resource<ShaderStage> * Stage = ResourceManager::Load<ShaderStage>(WStringToHash(LoadData.FragmentShader));
		if (Stage && Stage->GetData()) {
			Program->AppendStage(Stage->GetData());
		}
	}
	if (LoadData.ComputeShader.size() > 0) {
		Resource<ShaderStage> * Stage = ResourceManager::Load<ShaderStage>(WStringToHash(LoadData.ComputeShader));
		if (Stage && Stage->GetData()) {
			Program->AppendStage(Stage->GetData());
		}
	}
	if (LoadData.GeometryShader.size() > 0) {
		Resource<ShaderStage> * Stage = ResourceManager::Load<ShaderStage>(WStringToHash(LoadData.GeometryShader));
		if (Stage && Stage->GetData()) {
			Program->AppendStage(Stage->GetData());
		}
	}
	Program->Compile();

	Resource<ShaderProgram> * ResourceAdded = new Resource<ShaderProgram>(LoadData.Name, LoadData.GUID, Program);
	Resources.insert(std::pair<const size_t, BaseResource*>(ResourceAdded->GetIdentifier(), ResourceAdded));
	return ResourceAdded;
}

#endif // RESOURCES_ADD_SHADERPROGRAM