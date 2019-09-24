
#include "CoreMinimal.h"
#include "Resources/ShaderManager.h"
#include "Resources/TextureManager.h"

#include "Files/FileManager.h"
#include <yaml-cpp/yaml.h>

namespace YAML {
	template<>
	struct convert<EmptySource::EShaderUniformType> {
		static Node encode(const EmptySource::EShaderUniformType& rhs) {
			Node node;
			switch (rhs) {
				case EmptySource::EShaderUniformType::None:           node.push_back("None");           break;
				case EmptySource::EShaderUniformType::Matrix4x4Array: node.push_back("Matrix4x4Array"); break;
				case EmptySource::EShaderUniformType::Matrix4x4:      node.push_back("Matrix4x4");      break;
				case EmptySource::EShaderUniformType::FloatArray:     node.push_back("FloatArray");     break;
				case EmptySource::EShaderUniformType::Float:          node.push_back("Float");          break;
				case EmptySource::EShaderUniformType::Float2DArray:   node.push_back("Float2DArray");   break;
				case EmptySource::EShaderUniformType::Float2D:        node.push_back("Float2D");        break;
				case EmptySource::EShaderUniformType::Float3DArray:   node.push_back("Float3DArray");   break;
				case EmptySource::EShaderUniformType::Float3D:        node.push_back("Float3D");        break;
				case EmptySource::EShaderUniformType::Float4DArray:   node.push_back("Float4DArray");   break;
				case EmptySource::EShaderUniformType::Float4D:        node.push_back("Float4D");        break;
				case EmptySource::EShaderUniformType::Texture2D:      node.push_back("Texture2D");      break;
				case EmptySource::EShaderUniformType::Cubemap:        node.push_back("Cubemap");        break;
				case EmptySource::EShaderUniformType::Int:            node.push_back("Int");            break;
				case EmptySource::EShaderUniformType::IntArray:       node.push_back("IntArray");       break;
			}
			return node;
		}

		static bool decode(const Node& node, EmptySource::EShaderUniformType& rhs) {
			if (!node.IsSequence()) {
				return false;
			}
			EmptySource::NString TypeName = node.as<EmptySource::NString>();
			if (TypeName == "None")           rhs = EmptySource::EShaderUniformType::None;           return true;
			if (TypeName == "Matrix4x4Array") rhs = EmptySource::EShaderUniformType::Matrix4x4Array; return true;
			if (TypeName == "Matrix4x4")      rhs = EmptySource::EShaderUniformType::Matrix4x4;      return true;
			if (TypeName == "FloatArray")     rhs = EmptySource::EShaderUniformType::FloatArray;     return true;
			if (TypeName == "Float")          rhs = EmptySource::EShaderUniformType::Float;          return true;
			if (TypeName == "Float2DArray")   rhs = EmptySource::EShaderUniformType::Float2DArray;   return true;
			if (TypeName == "Float2D")        rhs = EmptySource::EShaderUniformType::Float2D;        return true;
			if (TypeName == "Float3DArray")   rhs = EmptySource::EShaderUniformType::Float3DArray;   return true;
			if (TypeName == "Float3D")        rhs = EmptySource::EShaderUniformType::Float3D;        return true;
			if (TypeName == "Float4DArray")   rhs = EmptySource::EShaderUniformType::Float4DArray;   return true;
			if (TypeName == "Float4D")        rhs = EmptySource::EShaderUniformType::Float4D;        return true;
			if (TypeName == "Texture2D")      rhs = EmptySource::EShaderUniformType::Texture2D;      return true;
			if (TypeName == "Cubemap")        rhs = EmptySource::EShaderUniformType::Cubemap;        return true;
			if (TypeName == "Int")            rhs = EmptySource::EShaderUniformType::Int;            return true;
			if (TypeName == "IntArray")       rhs = EmptySource::EShaderUniformType::IntArray;       return true;
			return true;
		}
	};
}

namespace EmptySource {

	RShaderPtr ShaderManager::GetProgram(const IName & Name) const {
		return GetProgram(Name.GetID());
	}

	RShaderPtr ShaderManager::GetProgram(const size_t & UID) const {
		auto Resource = ShaderProgramList.find(UID);
		if (Resource != ShaderProgramList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	void ShaderManager::FreeShaderProgram(const IName & Name) {
		size_t UID = Name.GetID();
		ShaderNameList.erase(UID);
		ShaderProgramList.erase(UID);
	}

	void ShaderManager::AddShaderProgram(RShaderPtr & Shader) {
		size_t UID = Shader->GetName().GetID();
		ShaderNameList.insert({ UID, Shader->GetName() });
		ShaderProgramList.insert({ UID, Shader });
	}

	TArray<IName> ShaderManager::GetResourceShaderNames() const {
		TArray<IName> Names;
		for (auto KeyValue : ShaderNameList)
			Names.push_back(KeyValue.second);
		std::sort(Names.begin(), Names.end(), [](const IName& First, const IName& Second) {
			return First < Second;
		});
		return Names;
	}

	void ShaderManager::LoadResourcesFromFile(const WString & FilePath) {
		FileStream * ResourcesFile = ResourceManager::GetResourcesFile(FilePath);

		YAML::Node BaseNode; {
			NString FileInfo;
			if (ResourcesFile == NULL || !ResourcesFile->ReadNarrowStream(&FileInfo))
				return;
			try {
				BaseNode = YAML::Load(FileInfo.c_str());
			} catch (...) {
				return;
			}
			ResourcesFile->Close();
		}

		YAML::Node ResourcesNode = BaseNode["Resources"];

		if (ResourcesNode.IsDefined()) {
			for (size_t i = 0; i < ResourcesNode.size(); i++) {
				if (ResourcesNode[i]["GUID"].IsDefined() && ResourcesNode[i]["ShaderProgram"].IsDefined()) {
					YAML::Node ShaderProgramNode = ResourcesNode[i]["ShaderProgram"];
					WString Name = ShaderProgramNode["Name"].IsDefined() ? Text::NarrowToWide(ShaderProgramNode["Name"].as<NString>()) : L"";
					WString OriginFile = ShaderProgramNode["OriginFile"].IsDefined() ? Text::NarrowToWide(ShaderProgramNode["OriginFile"].as<NString>()) : L"";
					CreateProgram(Name, OriginFile)->Load();
				}
			}
		}
	}

	TArray<ShaderManager::ShaderStageCode> ShaderManager::GetStagesCodeFromSource(const NString & Source) {
		YAML::Node BaseNode; {
			try {
				BaseNode = YAML::Load(Source.c_str());
			} catch (...) {
				return TArray<std::pair<EShaderStageType, NString>>();
			}
		}

		///// TODO: This may be retrieved from the renderer in the future
		YAML::Node SourceNode = BaseNode["GLSL"];

		TArray<std::pair<EShaderStageType, NString>> Stages;
		if (SourceNode.IsDefined()) {
			YAML::Node StagesNode = SourceNode["Stages"];
			if (StagesNode.IsDefined()) {
				for (YAML::iterator Iterator = StagesNode.begin(); Iterator != StagesNode.end(); ++Iterator) {
					const YAML::Node& Stage = *Iterator;
					NString Type = Stage["StageType"].as<NString>();

					if (Stage["Code"].IsDefined()) {
						std::pair<EShaderStageType, NString> StagePair(StringToStageType(Type), Stage["Code"].as<NString>());
						Stages.push_back(StagePair);
					}
				}
			}
		}
		return Stages;
	}

	RShaderPtr ShaderManager::CreateProgram(const WString & Name, const WString & Origin, const NString& Source) {
		RShaderPtr Shader = GetProgram(Name);
		if (Shader == NULL) {
			Shader = RShaderPtr(new RShader(Name, Origin, ""));
			AddShaderProgram(Shader);
		}
		return Shader;
	}

	ShaderManager & ShaderManager::GetInstance() {
		static ShaderManager Manager;
		return Manager;
	}

	NString ShaderManager::StageTypeToString(const EShaderStageType & Type) {
		switch (Type) {
		case ST_Vertex: return "Vertex";
		case ST_Pixel: return "Pixel";
		case ST_Geometry: return "Geometry";
		case ST_Compute: return "Compute";
		default: return "Unknown";
		}
	}

	EShaderStageType ShaderManager::StringToStageType(const NString & Type) {
		if (Type == "Vertex") return ST_Vertex;
		else if (Type == "Pixel") return ST_Pixel;
		else if (Type == "Geometry") return ST_Geometry;
		else if (Type == "Compute") return ST_Compute;
		return ST_Unknown;
	}

}
