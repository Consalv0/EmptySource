
#define RESOURCES_ADD_SHADERPROGRAM
#include "../include/ResourceShaderProgram.h"
#include "../include/ShaderProgram.h"

#include "../External/YAML/include/yaml-cpp/yaml.h"

namespace YAML {
	template<>
	struct convert<ShaderProgramData> {
		static Node encode(const ShaderProgramData& Element) {
			Node Node;
			Node["GUID"] = Element.GUID;
			YAML::Node ShaderProgramNode;
			ShaderProgramNode["Name"] = WStringToString(Element.Name);
			ShaderProgramNode["VertexShader"]   = WStringToString(Element.VertexShader);
			ShaderProgramNode["FragmentShader"] = WStringToString(Element.FragmentShader);
			ShaderProgramNode["ComputeShader"]  = WStringToString(Element.ComputeShader);
			ShaderProgramNode["GeometryShader"] = WStringToString(Element.GeometryShader);
			Node["ShaderProgram"] = ShaderProgramNode;
			return Node;
		}

		static bool decode(const Node& Node, ShaderProgramData& Element) {
			if (!Node["ShaderProgram"].IsDefined()) {
				return false;
			}

			Element.GUID = Node["GUID"].as<size_t>();
			Element.Name = Node["ShaderProgram"]["Name"].IsDefined() ? StringToWString(Node["ShaderProgram"]["Name"].as<String>()) : L"";
			Element.VertexShader = Node["ShaderProgram"]["VertexShader"].IsDefined() ?
				StringToWString(Node["ShaderProgram"]["VertexShader"].as<String>()) : L"";
			Element.FragmentShader = Node["ShaderProgram"]["FragmentShader"].IsDefined() ?
				StringToWString(Node["ShaderProgram"]["FragmentShader"].as<String>()) : L"";
			Element.ComputeShader = Node["ShaderProgram"]["ComputeShader"].IsDefined() ?
				StringToWString(Node["ShaderProgram"]["ComputeShader"].as<String>()) : L"";
			Element.GeometryShader = Node["ShaderProgram"]["GeometryShader"].IsDefined() ?
				StringToWString(Node["ShaderProgram"]["GeometryShader"].as<String>()) : L"";

			return true;
		}
	};
}

template<>
bool ResourceManager::GetResourceData<ShaderProgramData>(const WString & Name, ShaderProgramData & ResourceData) {
	FileStream * ResourcesFile = GetResourcesFile();
	YAML::Node BaseNode;
	bool bNeedsModification = false;
	{
		String FileInfo;
		if (ResourcesFile == NULL || !ResourcesFile->ReadNarrowStream(&FileInfo))
			return false;
		BaseNode = YAML::Load(FileInfo.c_str()); 
	}

	YAML::Node ResourcesNode = BaseNode["Resources"];
	int FileNodePos = -1;

	if (ResourcesNode.IsDefined()) {
		String FileString = WStringToString(Name);
		for (size_t i = 0; i < ResourcesNode.size(); i++) {
			if (ResourcesNode[i]["ShaderProgram"].IsDefined() && ResourcesNode[i]["ShaderProgram"]["Name"].IsDefined()
				&& Text::CompareIgnoreCase(ResourcesNode[i]["ShaderProgram"]["Name"].as<String>(), FileString)) {
				FileNodePos = (int)i;
				break;
			}
		}
		YAML::Node FileNode;
		if (FileNodePos < 0) {
			FileNodePos = (int)ResourcesNode.size();
			ResourcesNode.push_back(FileNode);
			bNeedsModification = true;
		}
		FileNode = ResourcesNode[FileNodePos];
		if (FileNode["GUID"].IsNull()) {
			FileNode["GUID"] = WStringToHash(Name);
			bNeedsModification = true;
		}
		if (!FileNode["ShaderProgram"].IsDefined()) {
			YAML::Node ShaderProgramNode;
			ShaderProgramNode["Name"] = FileString;
			FileNode["ShaderProgram"] = ShaderProgramNode;
			bNeedsModification = true;
		}
		else {
			if (!FileNode["ShaderProgram"]["Name"].IsDefined()) {
				FileNode["ShaderProgram"]["Name"] = FileString;
				bNeedsModification = true;
			}
		}
	}
	else {
		return false;
	}

	ResourceData = ResourcesNode[FileNodePos].as<ShaderProgramData>();

	if (bNeedsModification && ResourcesFile->IsValid()) {
		ResourcesFile->Clean();
		YAML::Emitter Out;
		Out << BaseNode;
		(*ResourcesFile) << Out.c_str();
		ResourcesFile->Close();
	}

	return true;
}

template<>
bool ResourceManager::GetResourceData<ShaderProgramData>(const size_t & GUID, ShaderProgramData & ResourceData) {
	FileStream * ResourcesFile = GetResourcesFile();
	YAML::Node BaseNode;
	{
		String FileInfo;
		if (ResourcesFile == NULL || !ResourcesFile->ReadNarrowStream(&FileInfo))
			return false;
		BaseNode = YAML::Load(FileInfo.c_str());
	}

	YAML::Node ResourcesNode = BaseNode["Resources"];
	int FileNodePos = -1;

	if (ResourcesNode.IsDefined()) {
		for (size_t i = 0; i < ResourcesNode.size(); i++) {
			if (ResourcesNode[i]["GUID"].IsDefined() && ResourcesNode[i]["GUID"].as<size_t>() == GUID) {
				FileNodePos = (int)i;
				break;
			}
		}
		if (FileNodePos < 0)
			return false;
	}
	else {
		return false;
	}

	ResourceData = ResourcesNode[FileNodePos].as<ShaderProgramData>();

	return true;
}