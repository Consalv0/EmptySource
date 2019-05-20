
#define RESOURCES_ADD_SHADERSTAGE
#include "../include/ResourceShaderStage.h"
#include "../include/ShaderStage.h"

#include "../External/YAML/include/yaml-cpp/yaml.h"

namespace YAML {
	template<>
	struct convert<ShaderStageData> {
		static Node encode(const ShaderStageData& Element) {
			Node Node;
			Node["GUID"] = Element.GUID;
			
			YAML::Node ShaderStageNode;
			if (Element.Type == Vertex)
				ShaderStageNode["Type"] = "Vertex";
			else if (Element.Type == Fragment)
				ShaderStageNode["Type"] = "Fragment";
			else if (Element.Type == Geometry)
				ShaderStageNode["Type"] = "Geometry";
			else if (Element.Type == Compute)
				ShaderStageNode["Type"] = "Compute";

			ShaderStageNode["FilePath"] = WStringToString(Element.FilePath);
			Node["ShaderStage"] = ShaderStageNode;
			return Node;
		}

		static bool decode(const Node& Node, ShaderStageData& Element) {
			if (!Node["ShaderStage"].IsDefined()) {
				return false;
			}

			Element.GUID = Node["GUID"].as<size_t>();
			Element.FilePath = Node["ShaderStage"]["FilePath"].IsDefined() ? StringToWString(Node["ShaderStage"]["FilePath"].as<String>()) : L"";
			String Type = Node["ShaderStage"]["Type"].IsDefined() ? Node["ShaderStage"]["Type"].as<String>() : "Vertex";

			if (Type == "Vertex")
				Element.Type = Vertex;
			else if (Type == "Fragment")
				Element.Type = Fragment;
			else if (Type == "Geometry")
				Element.Type = Geometry;
			else if (Type == "Compute")
				Element.Type = Compute;

			return true;
		}
	};
}

template<>
bool ResourceManager::ResourceToList<ShaderStageData>(const WString & File, ShaderStageData & ResourceData) {
	static WString ResourceFile = L"Resources/Resouces.yaml";
	FileStream * InfoFile = FileManager::GetFile(ResourceFile);
	YAML::Node BaseNode;
	if (InfoFile == NULL) {
		InfoFile = FileManager::MakeFile(ResourceFile);
		YAML::Node ResourcesNode;
		BaseNode["Resources"] = ResourcesNode;
	}
	else {
		String FileInfo;
		if (!InfoFile->ReadNarrowStream(&FileInfo))
			return false;
		BaseNode = YAML::Load(FileInfo.c_str());
	}

	YAML::Node ResourcesNode = BaseNode["Resources"];
	int FileNodePos = -1;

	if (ResourcesNode.IsDefined()) {
		String FileString = WStringToString(File);
		for (size_t i = 0; i < ResourcesNode.size(); i++) {
			if (ResourcesNode[i]["ShaderStage"].IsDefined() && ResourcesNode[i]["ShaderStage"]["FilePath"].IsDefined()
				&& Text::CompareIgnoreCase(ResourcesNode[i]["ShaderStage"]["FilePath"].as<String>(), FileString))
			{
				FileNodePos = (int)i;
				break;
			}
		}
		YAML::Node FileNode;
		if (FileNodePos < 0) {
			FileNodePos = (int)ResourcesNode.size();
			ResourcesNode.push_back(FileNode);
		}
		FileNode = ResourcesNode[FileNodePos];
		FileNode["GUID"] = GetHashName(File);
		if (!FileNode["ShaderStage"].IsDefined()) {
			YAML::Node ShaderStageNode;
			ShaderStageNode["FilePath"] = FileString;
			ShaderStageNode["Type"] = "Vertex";
			FileNode["ShaderStage"] = ShaderStageNode;
		}
		else {
			if (!FileNode["ShaderStage"]["FilePath"].IsDefined()) {
				FileNode["ShaderStage"]["FilePath"] = FileString;
			}
			if (!FileNode["ShaderStage"]["Type"].IsDefined()) {
				FileNode["ShaderStage"]["Type"] = "Vertex";
			}
		}
	}

	ResourceData = ResourcesNode[FileNodePos].as<ShaderStageData>();

	if (InfoFile->IsValid()) {
		InfoFile->Clean();
		YAML::Emitter Out;
		Out << BaseNode;
		(*InfoFile) << Out.c_str();
		InfoFile->Close();
	}

	return true;
}
