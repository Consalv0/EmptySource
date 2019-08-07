#pragma once

#ifdef RESOURCES_ADD_SHADERPROGRAM
#define RESOURCES_ADD_SHADERSTAGE
#include "Resources/Resources.h"
#include "Rendering/ShaderProgram.h"

namespace EmptySource {

	struct ShaderProgramData {
		size_t GUID;
		WString Name;
		size_t VertexShader;
		size_t FragmentShader;
		size_t ComputeShader;
		size_t GeometryShader;
	};

	template<>
	bool OldResourceManager::GetResourceData<ShaderProgramData>(const WString & File, ShaderProgramData & ResourceData);

	template<>
	inline Resource<ShaderProgram> * OldResourceManager::Load(const WString & Name) {
		ShaderProgramData LoadData;
		if (!GetResourceData<ShaderProgramData>(Name, LoadData))
			return NULL;

		auto ResourceFind = Resources.find(LoadData.GUID);
		if (ResourceFind != Resources.end()) {
			return dynamic_cast<Resource<ShaderProgram>*>(ResourceFind->second);
		}

		ShaderProgram * Program = new ShaderProgram(LoadData.Name);
		if (LoadData.VertexShader > 0) {
			Resource<ShaderStage> * Stage = OldResourceManager::Load<ShaderStage>(LoadData.VertexShader);
			if (Stage && Stage->GetData()) {
				Program->AppendStage(Stage->GetData());
			}
		}
		if (LoadData.FragmentShader > 0) {
			Resource<ShaderStage> * Stage = OldResourceManager::Load<ShaderStage>(LoadData.FragmentShader);
			if (Stage && Stage->GetData()) {
				Program->AppendStage(Stage->GetData());
			}
		}
		if (LoadData.ComputeShader > 0) {
			Resource<ShaderStage> * Stage = OldResourceManager::Load<ShaderStage>(LoadData.ComputeShader);
			if (Stage && Stage->GetData()) {
				Program->AppendStage(Stage->GetData());
			}
		}
		if (LoadData.GeometryShader > 0) {
			Resource<ShaderStage> * Stage = OldResourceManager::Load<ShaderStage>(LoadData.GeometryShader);
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
	bool OldResourceManager::GetResourceData<ShaderProgramData>(const size_t & GUID, ShaderProgramData & ResourceData);

	template<>
	inline Resource<ShaderProgram> * OldResourceManager::Load(const size_t & GUID) {
		ShaderProgramData LoadData;
		if (!GetResourceData<ShaderProgramData>(GUID, LoadData))
			return NULL;

		auto ResourceFind = Resources.find(LoadData.GUID);
		if (ResourceFind != Resources.end()) {
			return dynamic_cast<Resource<ShaderProgram>*>(ResourceFind->second);
		}

		ShaderProgram * Program = new ShaderProgram(LoadData.Name);
		if (LoadData.VertexShader > 0) {
			Resource<ShaderStage> * Stage = OldResourceManager::Load<ShaderStage>(LoadData.VertexShader);
			if (Stage && Stage->GetData()) {
				Program->AppendStage(Stage->GetData());
			}
		}
		if (LoadData.FragmentShader > 0) {
			Resource<ShaderStage> * Stage = OldResourceManager::Load<ShaderStage>(LoadData.FragmentShader);
			if (Stage && Stage->GetData()) {
				Program->AppendStage(Stage->GetData());
			}
		}
		if (LoadData.ComputeShader > 0) {
			Resource<ShaderStage> * Stage = OldResourceManager::Load<ShaderStage>(LoadData.ComputeShader);
			if (Stage && Stage->GetData()) {
				Program->AppendStage(Stage->GetData());
			}
		}
		if (LoadData.GeometryShader > 0) {
			Resource<ShaderStage> * Stage = OldResourceManager::Load<ShaderStage>(LoadData.GeometryShader);
			if (Stage && Stage->GetData()) {
				Program->AppendStage(Stage->GetData());
			}
		}
		Program->Compile();

		Resource<ShaderProgram> * ResourceAdded = new Resource<ShaderProgram>(LoadData.Name, LoadData.GUID, Program);
		Resources.insert(std::pair<const size_t, BaseResource*>(ResourceAdded->GetIdentifier(), ResourceAdded));
		return ResourceAdded;
	}

}
#endif // RESOURCES_ADD_SHADERPROGRAM