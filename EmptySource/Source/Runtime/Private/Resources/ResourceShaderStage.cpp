
#define RESOURCES_ADD_SHADERSTAGE
#include "Resources/ResourceShaderStage.h"
#include "Graphics/ShaderStage.h"

#include "../External/YAML/include/yaml-cpp/yaml.h"

namespace YAML {

	using namespace EmptySource;

	template<>
	struct convert<ShaderStageData> {
		static Node encode(const ShaderStageData& Element) {
			Node Node;
			Node["GUID"] = Element.GUID;
			
			YAML::Node ShaderStageNode;
			if (Element.Type == ST_Vertex)
				ShaderStageNode["Type"] = "Vertex";
			else if (Element.Type == ST_Fragment)
				ShaderStageNode["Type"] = "Fragment";
			else if (Element.Type == ST_Geometry)
				ShaderStageNode["Type"] = "Geometry";
			else if (Element.Type == ST_Compute)
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
				Element.Type = ST_Vertex;
			else if (Type == "Fragment")
				Element.Type = ST_Fragment;
			else if (Type == "Geometry")
				Element.Type = ST_Geometry;
			else if (Type == "Compute")
				Element.Type = ST_Compute;

			return true;
		}
	};

}

namespace EmptySource {

	template<>
	bool OldResourceManager::GetResourceData<ShaderStageData>(const WString & File, ShaderStageData & ResourceData) {
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
				bNeedsModification = true;
			}
			FileNode = ResourcesNode[FileNodePos];
			if (!FileNode["GUID"].IsDefined() || FileNode["GUID"].IsNull()) {
				FileNode["GUID"] = WStringToHash(File);
				bNeedsModification = true;
			}
			if (!FileNode["ShaderStage"].IsDefined()) {
				YAML::Node ShaderStageNode;
				ShaderStageNode["FilePath"] = FileString;
				ShaderStageNode["Type"] = "Vertex";
				FileNode["ShaderStage"] = ShaderStageNode;
				bNeedsModification = true;
			}
			else {
				if (!FileNode["ShaderStage"]["FilePath"].IsDefined()) {
					FileNode["ShaderStage"]["FilePath"] = FileString;
					bNeedsModification = true;
				}
				if (!FileNode["ShaderStage"]["Type"].IsDefined()) {
					FileNode["ShaderStage"]["Type"] = "Vertex";
					bNeedsModification = true;
				}
			}
		}

		ResourceData = ResourcesNode[FileNodePos].as<ShaderStageData>();

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
	bool OldResourceManager::GetResourceData<ShaderStageData>(const size_t & GUID, ShaderStageData & ResourceData) {
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

		ResourceData = ResourcesNode[FileNodePos].as<ShaderStageData>();

		return true;
	}

}