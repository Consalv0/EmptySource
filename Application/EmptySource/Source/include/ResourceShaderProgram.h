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
bool ResourceManager::ResourceToList<ShaderProgramData>(const WString & File, ShaderProgramData & ResourceData);

template<>
inline Resource<ShaderProgram> * ResourceManager::Load(const WString & Name) {
	ShaderProgramData LoadData;
	if (!ResourceToList<ShaderProgramData>(Name, LoadData))
		return NULL;

	auto ResourceFind = Resources.find(GetHashName(Name));
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

	Resource<ShaderProgram> * ResourceAdded = new Resource<ShaderProgram>(Name, Program);
	Resources.insert(std::pair<const size_t, BaseResource*>(ResourceAdded->GetIdentifierHash(), ResourceAdded));
	return ResourceAdded;
}
#endif // RESOURCES_ADD_SHADERPROGRAM