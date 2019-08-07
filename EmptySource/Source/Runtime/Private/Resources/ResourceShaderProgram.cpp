
#define RESOURCES_ADD_SHADERPROGRAM
#include "Resources/ResourceShaderProgram.h"
#include "Rendering/ShaderProgram.h"

#include <yaml-cpp/yaml.h>

namespace YAML {
	using namespace EmptySource;

	template<>
	struct convert<ShaderProgramData> {
		static Node encode(const ShaderProgramData& Element) {
			Node Node;
			Node["GUID"] = Element.GUID;
			YAML::Node ShaderProgramNode;
			ShaderProgramNode["Name"] = Text::WideToNarrow(Element.Name);
			ShaderProgramNode["VertexShader"]   = Element.VertexShader;
			ShaderProgramNode["FragmentShader"] = Element.FragmentShader;
			ShaderProgramNode["ComputeShader"]  = Element.ComputeShader;
			ShaderProgramNode["GeometryShader"] = Element.GeometryShader;
			Node["ShaderProgram"] = ShaderProgramNode;
			return Node;
		}

		static bool decode(const Node& Node, ShaderProgramData& Element) {
			if (!Node["ShaderProgram"].IsDefined()) {
				return false;
			}

			Element.GUID = Node["GUID"].as<size_t>();
			Element.Name = Node["ShaderProgram"]["Name"].IsDefined() ? Text::NarrowToWide(Node["ShaderProgram"]["Name"].as<NString>()) : L"";
			Element.VertexShader = Node["ShaderProgram"]["VertexShader"].IsDefined() ?
				Node["ShaderProgram"]["VertexShader"].as<size_t>() : 0;
			Element.FragmentShader = Node["ShaderProgram"]["FragmentShader"].IsDefined() ?
				Node["ShaderProgram"]["FragmentShader"].as<size_t>() : 0;
			Element.ComputeShader = Node["ShaderProgram"]["ComputeShader"].IsDefined() ?
				Node["ShaderProgram"]["ComputeShader"].as<size_t>() : 0;
			Element.GeometryShader = Node["ShaderProgram"]["GeometryShader"].IsDefined() ?
				Node["ShaderProgram"]["GeometryShader"].as<size_t>() : 0;

			return true;
		}
	};
}

namespace EmptySource {

	template<>
	bool OldResourceManager::GetResourceData<ShaderProgramData>(const WString & Name, ShaderProgramData & ResourceData) {
		FileStream * ResourcesFile = GetResourcesFile();
		YAML::Node BaseNode;
		bool bNeedsModification = false;
		{
			NString FileInfo;
			if (ResourcesFile == NULL || !ResourcesFile->ReadNarrowStream(&FileInfo))
				return false;
			BaseNode = YAML::Load(FileInfo.c_str());
		}

		YAML::Node ResourcesNode = BaseNode["Resources"];
		int FileNodePos = -1;

		if (ResourcesNode.IsDefined()) {
			NString FileString = Text::WideToNarrow(Name);
			for (size_t i = 0; i < ResourcesNode.size(); i++) {
				if (ResourcesNode[i]["ShaderProgram"].IsDefined() && ResourcesNode[i]["ShaderProgram"]["Name"].IsDefined()
					&& Text::CompareIgnoreCase(ResourcesNode[i]["ShaderProgram"]["Name"].as<NString>(), FileString)) {
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
			if (FileNode["GUID"].IsNull() || !FileNode["GUID"].IsDefined()) {
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
	bool OldResourceManager::GetResourceData<ShaderProgramData>(const size_t & GUID, ShaderProgramData & ResourceData) {
		FileStream * ResourcesFile = GetResourcesFile();
		YAML::Node BaseNode;
		{
			NString FileInfo;
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

}