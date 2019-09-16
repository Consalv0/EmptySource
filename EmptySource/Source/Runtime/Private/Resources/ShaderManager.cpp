
#include "CoreMinimal.h"
#include "Resources/ShaderManager.h"
#include "Resources/TextureManager.h"

#include <yaml-cpp/yaml.h>

namespace YAML {
	template<>
	struct convert<EmptySource::EShaderPropertyType> {
		static Node encode(const EmptySource::EShaderPropertyType& rhs) {
			Node node;
			switch (rhs) {
				case EmptySource::EShaderPropertyType::None:           node.push_back("None");           break;
				case EmptySource::EShaderPropertyType::Matrix4x4Array: node.push_back("Matrix4x4Array"); break;
				case EmptySource::EShaderPropertyType::Matrix4x4:      node.push_back("Matrix4x4");      break;
				case EmptySource::EShaderPropertyType::FloatArray:     node.push_back("FloatArray");     break;
				case EmptySource::EShaderPropertyType::Float:          node.push_back("Float");          break;
				case EmptySource::EShaderPropertyType::Float2DArray:   node.push_back("Float2DArray");   break;
				case EmptySource::EShaderPropertyType::Float2D:        node.push_back("Float2D");        break;
				case EmptySource::EShaderPropertyType::Float3DArray:   node.push_back("Float3DArray");   break;
				case EmptySource::EShaderPropertyType::Float3D:        node.push_back("Float3D");        break;
				case EmptySource::EShaderPropertyType::Float4DArray:   node.push_back("Float4DArray");   break;
				case EmptySource::EShaderPropertyType::Float4D:        node.push_back("Float4D");        break;
				case EmptySource::EShaderPropertyType::Texture2D:      node.push_back("Texture2D");      break;
				case EmptySource::EShaderPropertyType::Cubemap:        node.push_back("Cubemap");        break;
				case EmptySource::EShaderPropertyType::Int:            node.push_back("Int");            break;
				case EmptySource::EShaderPropertyType::IntArray:       node.push_back("IntArray");       break;
			}
			return node;
		}

		static bool decode(const Node& node, EmptySource::EShaderPropertyType& rhs) {
			if (!node.IsSequence()) {
				return false;
			}
			EmptySource::NString TypeName = node.as<EmptySource::NString>();
			if (TypeName == "None")           rhs = EmptySource::EShaderPropertyType::None;           return true;
			if (TypeName == "Matrix4x4Array") rhs = EmptySource::EShaderPropertyType::Matrix4x4Array; return true;
			if (TypeName == "Matrix4x4")      rhs = EmptySource::EShaderPropertyType::Matrix4x4;      return true;
			if (TypeName == "FloatArray")     rhs = EmptySource::EShaderPropertyType::FloatArray;     return true;
			if (TypeName == "Float")          rhs = EmptySource::EShaderPropertyType::Float;          return true;
			if (TypeName == "Float2DArray")   rhs = EmptySource::EShaderPropertyType::Float2DArray;   return true;
			if (TypeName == "Float2D")        rhs = EmptySource::EShaderPropertyType::Float2D;        return true;
			if (TypeName == "Float3DArray")   rhs = EmptySource::EShaderPropertyType::Float3DArray;   return true;
			if (TypeName == "Float3D")        rhs = EmptySource::EShaderPropertyType::Float3D;        return true;
			if (TypeName == "Float4DArray")   rhs = EmptySource::EShaderPropertyType::Float4DArray;   return true;
			if (TypeName == "Float4D")        rhs = EmptySource::EShaderPropertyType::Float4D;        return true;
			if (TypeName == "Texture2D")      rhs = EmptySource::EShaderPropertyType::Texture2D;      return true;
			if (TypeName == "Cubemap")        rhs = EmptySource::EShaderPropertyType::Cubemap;        return true;
			if (TypeName == "Int")            rhs = EmptySource::EShaderPropertyType::Int;            return true;
			if (TypeName == "IntArray")       rhs = EmptySource::EShaderPropertyType::IntArray;       return true;
			return true;
		}
	};
}

namespace EmptySource {

	ShaderPtr ShaderManager::GetProgram(const WString & Name) const {
		size_t UID = WStringToHash(Name);
		return GetProgram(UID);
	}

	ShaderPtr ShaderManager::GetProgram(const size_t & UID) const {
		auto Resource = ShaderProgramList.find(UID);
		if (Resource != ShaderProgramList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	ShaderStagePtr ShaderManager::GetStage(const WString & Name) const {
		size_t UID = WStringToHash(Name);
		return GetStage(UID);
	}

	ShaderStagePtr ShaderManager::GetStage(const size_t & UID) const {
		auto Resource = ShaderStageList.find(UID);
		if (Resource != ShaderStageList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	void ShaderManager::FreeShaderProgram(const WString & Name) {
		size_t UID = WStringToHash(Name);
		ShaderNameList.erase(UID);
		ShaderProgramList.erase(UID);
	}

	void ShaderManager::FreeShaderStage(const WString & Name) {
		size_t UID = WStringToHash(Name);
		ShaderStageNameList.erase(UID);
		ShaderStageList.erase(UID);
	}

	void ShaderManager::AddShaderProgram(ShaderPtr & Shader) {
		size_t UID = WStringToHash(Shader->GetName());
		ShaderNameList.insert({ UID, Shader->GetName() });
		ShaderProgramList.insert({ UID, Shader });
	}

	void ShaderManager::AddShaderStage(const WString & Name, ShaderStagePtr & Stage) {
		size_t UID = WStringToHash(Name);
		ShaderStageNameList.insert({ UID, Name });
		ShaderStageList.insert({ UID, Stage });
	}

	TArray<WString> ShaderManager::GetResourceShaderNames() const {
		TArray<WString> Names;
		for (auto KeyValue : ShaderNameList)
			Names.push_back(KeyValue.second);
		std::sort(Names.begin(), Names.end(), [](const WString& first, const WString& second) {
			unsigned int i = 0;
			while ((i < first.length()) && (i < second.length())) {
				if (tolower(first[i]) < tolower(second[i])) return true;
				else if (tolower(first[i]) > tolower(second[i])) return false;
				++i;
			}
			return (first.length() < second.length());
		});
		return Names;
	}

	TArray<WString> ShaderManager::GetResourceShaderStageNames() const {
		TArray<WString> Names;
		for (auto KeyValue : ShaderStageNameList)
			Names.push_back(KeyValue.second);
		std::sort(Names.begin(), Names.end(), [](const WString& first, const WString& second) {
			unsigned int i = 0;
			while ((i < first.length()) && (i < second.length())) {
				if (tolower(first[i]) < tolower(second[i])) return true;
				else if (tolower(first[i]) > tolower(second[i])) return false;
				++i;
			}
			return (first.length() < second.length());
		});
		return Names;
	}

	void ShaderManager::LoadResourcesFromFile(const WString & FilePath) {
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
						Stages.push_back(GetStage(ShaderProgramNode["VertexShader"].as<size_t>()));
					if (ShaderProgramNode["FragmentShader"].IsDefined())
						Stages.push_back(GetStage(ShaderProgramNode["FragmentShader"].as<size_t>()));
					if (ShaderProgramNode["ComputeShader"].IsDefined())
						Stages.push_back(GetStage(ShaderProgramNode["ComputeShader"].as<size_t>()));
					if (ShaderProgramNode["GeometryShader"].IsDefined())
						Stages.push_back(GetStage(ShaderProgramNode["GeometryShader"].as<size_t>()));

					YAML::Node PropertiesNode = ShaderProgramNode["Properties"];
					TArray<ShaderProperty> Properties;
					for (auto & Uniform : PropertiesNode) {
						NString UniformName = Uniform["Uniform"].as<NString>();
						NString TypeName    = Uniform["Type"].as<NString>();
						int Flags = SPFlags_None;
						if (Uniform["IsColor"]) {
							Flags |= Uniform["IsColor"].as<bool>() ? SPFlags_IsColor : SPFlags_None;
						}
						if      (TypeName == "None")           Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(), Flags);
						else if (TypeName == "Matrix4x4Array") Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<Matrix4x4>()), Flags);
						else if (TypeName == "Matrix4x4")      Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(Matrix4x4()), Flags);
						else if (TypeName == "FloatArray")     Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<float>()), Flags);
						else if (TypeName == "Float")          Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(Uniform["DefaultValue"].as<float>()), Flags);
						else if (TypeName == "Float2DArray")   Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<Vector2>()), Flags);
						else if (TypeName == "Float2D")        Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(Vector2(
							Uniform["DefaultValue"][0].as<float>(), Uniform["DefaultValue"][1].as<float>())
						), Flags);
						else if (TypeName == "Float3DArray")   Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<Vector3>()), Flags);
						else if (TypeName == "Float3D")        Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(Vector3(
							Uniform["DefaultValue"][0].as<float>(), Uniform["DefaultValue"][1].as<float>(), Uniform["DefaultValue"][2].as<float>())
						), Flags);
						else if (TypeName == "Float4DArray")   Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<Vector4>()), Flags);
						else if (TypeName == "Float4D")        Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(Vector4(
							Uniform["DefaultValue"][0].as<float>(), Uniform["DefaultValue"][1].as<float>(), Uniform["DefaultValue"][2].as<float>(),
							Uniform["DefaultValue"][3].as<float>())
						), Flags);
						else if (TypeName == "Texture2D")      Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(ETextureDimension::Texture2D,
							TextureManager::GetInstance().GetTexture(Text::NarrowToWide(Uniform["DefaultValue"].as<NString>()))
						), Flags);
						else if (TypeName == "Cubemap")        Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(ETextureDimension::Cubemap,
							NULL), Flags);
						else if (TypeName == "IntArray")       Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<int>()), Flags);
						else if (TypeName == "Int")            Properties.emplace_back(UniformName, ShaderProperty::PropertyValue(0), Flags);
					}

					ShaderPtr CreatedShader = ShaderProgram::Create(Name, Stages);
					CreatedShader->SetProperties(Properties);
					AddShaderProgram(CreatedShader);

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
