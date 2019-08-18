
#include "CoreMinimal.h"
#include "Resources/ResourceShader.h"

#include <yaml-cpp/yaml.h>

namespace EmptySource {

	void ResourceHolderShaderStage::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		if (Data.Code.empty())
			Stage = ShaderStage::CreateFromText(Data.Code, Data.ShaderType);
		else
			Stage = ShaderStage::CreateFromFile(Data.FilePath, Data.ShaderType);
		LoadState = LS_Loaded;
	}

	void ResourceHolderShaderStage::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		Stage.reset();
		LoadState = LS_Unloaded;
	}

	void ResourceHolderShaderStage::Reload() {
		Unload();
		Load();
	}

	ResourceHolderShaderStage::ResourceHolderShaderStage(const ResourceShaderStageData & InData)
		: ResourceHolder(InData.Name), Stage(), LoadState(LS_Unloaded) {
		Data.Name = InData.Name;
		Data.ShaderType = InData.ShaderType;
		Data.Code = InData.Code;
	}

	// Shader Program

	void ResourceHolderShader::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		TArray<ShaderStagePtr> Stages;
		if (Data.VertexShader != NULL && Data.VertexShader->IsValid())
			Stages.push_back(Data.VertexShader);
		if (Data.PixelShader != NULL && Data.PixelShader->IsValid())
			Stages.push_back(Data.PixelShader);
		if (Data.GeometryShader != NULL && Data.GeometryShader->IsValid())
			Stages.push_back(Data.GeometryShader);
		if (Data.ComputeShader != NULL && Data.ComputeShader->IsValid())
			Stages.push_back(Data.ComputeShader);

		LoadState = LS_Loading;
		Shader = ShaderProgram::Create(Name, Stages);
		LoadState = LS_Loaded;
	}

	void ResourceHolderShader::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		Shader.reset();
		LoadState = LS_Unloaded;
	}

	void ResourceHolderShader::Reload() {
		Unload();
		Load();
	}

	ResourceHolderShader::ResourceHolderShader(const ResourceShaderData & Data)
		: ResourceHolder(Data.Name), Data(Data), Shader(), LoadState(LS_Unloaded) {
	}

	ResourceHolderShader::ResourceHolderShader(const WString& Name, TArray<ShaderStagePtr> Stages) 
		: ResourceHolder(Name), Data(), Shader(), LoadState(LS_Unloaded) {
		Data.Name = Name;
		for (auto Stage : Stages) {
			switch (Stage->GetType()) {
			case ST_Vertex:
				Data.VertexShader = Stage;
			case ST_Compute:
				Data.ComputeShader = Stage;
			case ST_Pixel:
				Data.PixelShader = Stage;
			case ST_Geometry:
				Data.GeometryShader = Stage;
			default:
				break;
			}
		}
	}

	ResourceShaderData::ResourceShaderData(const WString& FilePath, size_t UID) : ResourceFile(FilePath) {
		FileStream * ResourcesFile = ResourceManager::GetResourcesFile(FilePath);
		YAML::Node BaseNode;
		{
			NString FileInfo;
			if (ResourcesFile == NULL || !ResourcesFile->ReadNarrowStream(&FileInfo))
				return;
			BaseNode = YAML::Load(FileInfo.c_str());
		}

		YAML::Node ResourcesNode = BaseNode["Resources"];

		int FileNodePos = -1;
		if (ResourcesNode.IsDefined()) {
			for (size_t i = 0; i < ResourcesNode.size(); i++) {
				if (ResourcesNode[i]["GUID"].IsDefined() && ResourcesNode[i]["GUID"].as<size_t>() == UID) {
					FileNodePos = (int)i;
					break;
				}
			}
			if (FileNodePos < 0)
				return;
		}
		else return;

		YAML::Node ShaderProgramNode = ResourcesNode[FileNodePos]["ShaderProgram"];
		Name = ShaderProgramNode["Name"].IsDefined() ? Text::NarrowToWide(ShaderProgramNode["Name"].as<NString>()) : L"";
		// VertexShader = ShaderProgramNode["VertexShader"].IsDefined() ?
		// 	ResourceManager::GetInstance().CreateResource<ResourceHolderShaderStage>(
		// 		ResourceShaderStageData(ResourceFile, ShaderProgramNode["VertexShader"].as<size_t>())) : NULL;
		// FragmentShader = ShaderProgramNode["FragmentShader"].IsDefined() ?
		// 	ResourceManager::GetInstance().CreateResource<ResourceHolderShaderStage>(
		// 		ResourceShaderStageData(ResourceFile, ShaderProgramNode["FragmentShader"].as<size_t>())) : NULL;
		// ComputeShader = ShaderProgramNode["ComputeShader"].IsDefined() ?
		// 	ResourceManager::GetInstance().CreateResource<ResourceHolderShaderStage>(
		// 		ResourceShaderStageData(ResourceFile, ShaderProgramNode["ComputeShader"].as<size_t>())) : NULL;
		// GeometryShader = ShaderProgramNode["GeometryShader"].IsDefined() ?
		// 	ResourceManager::GetInstance().CreateResource<ResourceHolderShaderStage>(
		// 		ResourceShaderStageData(ResourceFile, ShaderProgramNode["GeometryShader"].as<size_t>())) : NULL;

	}

	ResourceShaderStageData::ResourceShaderStageData(const WString & FilePath, size_t UID) : ResourceFile(FilePath) {
		FileStream * ResourcesFile = ResourceManager::GetResourcesFile(FilePath);
		YAML::Node BaseNode;
		{
			NString FileInfo;
			if (ResourcesFile == NULL || !ResourcesFile->ReadNarrowStream(&FileInfo))
				return;
			BaseNode = YAML::Load(FileInfo.c_str());
		}

		YAML::Node ResourcesNode = BaseNode["Resources"];

		int FileNodePos = -1;
		if (ResourcesNode.IsDefined()) {
			for (size_t i = 0; i < ResourcesNode.size(); i++) {
				if (ResourcesNode[i]["GUID"].IsDefined() && ResourcesNode[i]["GUID"].as<size_t>() == UID) {
					FileNodePos = (int)i;
					break;
				}
			}
			if (FileNodePos < 0)
				return;
		}
		else return;

		YAML::Node ShaderStageNode = ResourcesNode[FileNodePos]["ShaderStage"];
		this->FilePath = ShaderStageNode["FilePath"].IsDefined() ? Text::NarrowToWide(ShaderStageNode["FilePath"].as<NString>()) : L"";
		NString Type = ShaderStageNode["Type"].IsDefined() ? ShaderStageNode["Type"].as<NString>() : "Vertex";

		if (Type == "Vertex")
			ShaderType = ST_Vertex;
		else if (Type == "Pixel")
			ShaderType = ST_Pixel;
		else if (Type == "Geometry")
			ShaderType = ST_Geometry;
		else if (Type == "Compute")
			ShaderType = ST_Compute;

	}

}