
#include "CoreMinimal.h"
#include "Resources/ShaderManager.h"

#include <yaml-cpp/yaml.h>

namespace EmptySource {

	ShaderPtr ShaderManager::GetProgramByName(const WString & Name) const {
		size_t UID = WStringToHash(Name);
		return GetProgramByUniqueID(UID);
	}

	ShaderPtr ShaderManager::GetProgramByUniqueID(const size_t & UID) const {
		auto Resource = ShaderProgramList.find(UID);
		if (Resource != ShaderProgramList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	ShaderStagePtr ShaderManager::GetStageByName(const WString & Name) const {
		size_t UID = WStringToHash(Name);
		return GetStageByUniqueID(UID);
	}

	ShaderStagePtr ShaderManager::GetStageByUniqueID(const size_t & UID) const {
		auto Resource = ShaderStageList.find(UID);
		if (Resource != ShaderStageList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	void ShaderManager::AddShaderProgram(ShaderPtr & Shader) {
		size_t UID = WStringToHash(Shader->GetName());
		ShaderProgramList.insert({ UID, Shader });
	}

	void ShaderManager::AddShaderStage(const WString & Name, ShaderStagePtr & Stage) {
		size_t UID = WStringToHash(Name);
		ShaderStageList.insert({ UID, Stage });
	}

	void ShaderManager::GetResourcesFromFile(const WString & FilePath) {
		FileStream * ResourcesFile = ResourceManager::GetResourcesFile(FilePath);

		YAML::Node BaseNode; 
		{
			NString FileInfo;
			if (ResourcesFile == NULL || !ResourcesFile->ReadNarrowStream(&FileInfo))
				return;
			BaseNode = YAML::Load(FileInfo.c_str());
			ResourcesFile->Close();
		}

		YAML::Node ResourcesNode = BaseNode["Resources"];

		if (ResourcesNode.IsDefined()) {
			int FileNodePos = -1;

			for (size_t i = 0; i < ResourcesNode.size(); i++) {
				if (ResourcesNode[i]["GUID"].IsDefined() && ResourcesNode[i]["ShaderStage"].IsDefined()) {
					FileNodePos = (int)i;

					YAML::Node ShaderStageNode = ResourcesNode[FileNodePos]["ShaderStage"];
					WString FilePath = ShaderStageNode["FilePath"].IsDefined() ? Text::NarrowToWide(ShaderStageNode["FilePath"].as<NString>()) : L"";
					NString Type = ShaderStageNode["Type"].IsDefined() ? ShaderStageNode["Type"].as<NString>() : "Vertex";
					EShaderType ShaderType;
					if (Type == "Vertex")
						ShaderType = ST_Vertex;
					else if (Type == "Pixel")
						ShaderType = ST_Pixel;
					else if (Type == "Geometry")
						ShaderType = ST_Geometry;
					else if (Type == "Compute")
						ShaderType = ST_Compute;

					AddShaderStage(FilePath, ShaderStage::CreateFromFile(FilePath, ShaderType));
				}
			}

			if (FileNodePos < 0)
				return;

			for (size_t i = 0; i < ResourcesNode.size(); i++) {
				if (ResourcesNode[i]["GUID"].IsDefined() && ResourcesNode[i]["ShaderProgram"].IsDefined()) {
					FileNodePos = (int)i;

					YAML::Node ShaderProgramNode = ResourcesNode[FileNodePos]["ShaderProgram"];
					WString Name = ShaderProgramNode["Name"].IsDefined() ? Text::NarrowToWide(ShaderProgramNode["Name"].as<NString>()) : L"";
					TArray<ShaderStagePtr> Stages;
					if (ShaderProgramNode["VertexShader"].IsDefined()) 
						Stages.push_back(GetStageByUniqueID(ShaderProgramNode["VertexShader"].as<size_t>()));
					if (ShaderProgramNode["FragmentShader"].IsDefined())
						Stages.push_back(GetStageByUniqueID(ShaderProgramNode["FragmentShader"].as<size_t>()));
					if (ShaderProgramNode["ComputeShader"].IsDefined())
						Stages.push_back(GetStageByUniqueID(ShaderProgramNode["ComputeShader"].as<size_t>()));
					if (ShaderProgramNode["GeometryShader"].IsDefined())
						Stages.push_back(GetStageByUniqueID(ShaderProgramNode["GeometryShader"].as<size_t>()));

					AddShaderProgram(ShaderProgram::Create(Name, Stages));
				}
			}
		}
		else return;
	}

	ShaderManager & ShaderManager::GetInstance() {
		static ShaderManager Manager;
		return Manager;
	}

}
