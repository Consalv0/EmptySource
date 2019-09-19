
#include "CoreMinimal.h"
#include "Resources/ShaderProgram.h"

#include <yaml-cpp/yaml.h>

namespace EmptySource {

	bool RShaderStage::IsValid() {
		return LoadState == LS_Loaded && StagePointer->IsValid();
	}
	
	void RShaderStage::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			FileStream * ShaderCode = FileManager::GetFile(Origin);
			if (ShaderCode == NULL) {
				LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
				return;
			}
			if (!ShaderCode->ReadNarrowStream(&SourceCode)) {
				ShaderCode->Close();
				LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
				return;
			}

			ShaderCode->Close();
			if (!SourceCode.empty())
				StagePointer = ShaderStage::CreateFromText(SourceCode, StageType);
		}
		LoadState = StagePointer != NULL ? LS_Loaded : LS_Unloaded;
	}

	void RShaderStage::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		delete StagePointer;
		StagePointer = NULL;
		LoadState = LS_Unloaded;
	}

	void RShaderStage::Reload() {
		Unload();
		Load();
	}

	RShaderStage::RShaderStage(const IName & Name, const WString & Origin, EShaderStageType Type, const NString & Code)
		: ResourceHolder(Name, Origin), StagePointer(NULL), SourceCode(Code), StageType(Type) {
	}

	RShaderStage::~RShaderStage() {
		Unload();
	}

	// Shader Program
	bool RShaderProgram::IsValid() {
		return LoadState == LS_Loaded && ShaderPointer->IsValid();
	}

	void RShaderProgram::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			TArray<ShaderStage *> Stages;
			if (VertexShader != NULL && VertexShader->IsValid())
				Stages.push_back(VertexShader->GetShaderStage());
			if (PixelShader != NULL && PixelShader->IsValid())
				Stages.push_back(PixelShader->GetShaderStage());
			if (GeometryShader != NULL && GeometryShader->IsValid())
				Stages.push_back(GeometryShader->GetShaderStage());
			if (ComputeShader != NULL && ComputeShader->IsValid())
				Stages.push_back(ComputeShader->GetShaderStage());

			ShaderPointer = ShaderProgram::Create(Stages);
		}
		LoadState = ShaderPointer != NULL ? LS_Loaded : LS_Unloaded;
	}

	void RShaderProgram::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		delete ShaderPointer;
		ShaderPointer = NULL;
		LoadState = LS_Unloaded;
	}

	void RShaderProgram::Reload() {
		Unload();
		if (VertexShader)   VertexShader->Load();
		if (PixelShader)    PixelShader->Load();
		if (GeometryShader) GeometryShader->Load();
		if (ComputeShader)  ComputeShader->Load();
		Load();
	}

	void RShaderProgram::SetProperties(const TArrayInitializer<ShaderProperty> & InProperties) {
		Properties = InProperties;
	}

	void RShaderProgram::SetProperties(const TArray<ShaderProperty>& InProperties) {
		Properties = InProperties;
	}

	RShaderProgram::RShaderProgram(const IName & Name, const WString & Origin, TArray<RShaderStagePtr>& Stages) 
		: ResourceHolder(Name, Origin), Properties() {
		for (RShaderStagePtr & Stage : Stages) {
			switch (Stage->GetShaderType()) {
				case ST_Vertex:   VertexShader   = Stage; break;
				case ST_Pixel:    PixelShader    = Stage; break;
				case ST_Geometry: GeometryShader = Stage; break;
				case ST_Compute:  ComputeShader  = Stage; break;
				default: break;
			}
		}
	}

	RShaderProgram::~RShaderProgram() {
		Unload();
	}

	ResourceShaderData::ResourceShaderData(const WString& FilePath, size_t UID) : Origin(FilePath) {
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

	ResourceShaderStageData::ResourceShaderStageData(const WString & FilePath, size_t UID) : Origin(FilePath) {
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