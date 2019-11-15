
#include "CoreMinimal.h"
#include "Resources/ShaderManager.h"
#include "Resources/TextureManager.h"

#include "Files/FileManager.h"
#include <yaml-cpp/yaml.h>

namespace YAML {
	template<>
	struct convert<ESource::EShaderUniformType> {
		static Node encode(const ESource::EShaderUniformType& rhs) {
			Node node;
			switch (rhs) {
				case ESource::EShaderUniformType::None:           node.push_back("None");           break;
				case ESource::EShaderUniformType::Matrix4x4Array: node.push_back("Matrix4x4Array"); break;
				case ESource::EShaderUniformType::Matrix4x4:      node.push_back("Matrix4x4");      break;
				case ESource::EShaderUniformType::FloatArray:     node.push_back("FloatArray");     break;
				case ESource::EShaderUniformType::Float:          node.push_back("Float");          break;
				case ESource::EShaderUniformType::Float2DArray:   node.push_back("Float2DArray");   break;
				case ESource::EShaderUniformType::Float2D:        node.push_back("Float2D");        break;
				case ESource::EShaderUniformType::Float3DArray:   node.push_back("Float3DArray");   break;
				case ESource::EShaderUniformType::Float3D:        node.push_back("Float3D");        break;
				case ESource::EShaderUniformType::Float4DArray:   node.push_back("Float4DArray");   break;
				case ESource::EShaderUniformType::Float4D:        node.push_back("Float4D");        break;
				case ESource::EShaderUniformType::Texture2D:      node.push_back("Texture2D");      break;
				case ESource::EShaderUniformType::Cubemap:        node.push_back("Cubemap");        break;
				case ESource::EShaderUniformType::Int:            node.push_back("Int");            break;
				case ESource::EShaderUniformType::IntArray:       node.push_back("IntArray");       break;
			}
			return node;
		}

		static bool decode(const Node& node, ESource::EShaderUniformType& rhs) {
			if (!node.IsSequence()) {
				return false;
			}
			ESource::NString TypeName = node.as<ESource::NString>();
			if (TypeName == "None")           rhs = ESource::EShaderUniformType::None;           return true;
			if (TypeName == "Matrix4x4Array") rhs = ESource::EShaderUniformType::Matrix4x4Array; return true;
			if (TypeName == "Matrix4x4")      rhs = ESource::EShaderUniformType::Matrix4x4;      return true;
			if (TypeName == "FloatArray")     rhs = ESource::EShaderUniformType::FloatArray;     return true;
			if (TypeName == "Float")          rhs = ESource::EShaderUniformType::Float;          return true;
			if (TypeName == "Float2DArray")   rhs = ESource::EShaderUniformType::Float2DArray;   return true;
			if (TypeName == "Float2D")        rhs = ESource::EShaderUniformType::Float2D;        return true;
			if (TypeName == "Float3DArray")   rhs = ESource::EShaderUniformType::Float3DArray;   return true;
			if (TypeName == "Float3D")        rhs = ESource::EShaderUniformType::Float3D;        return true;
			if (TypeName == "Float4DArray")   rhs = ESource::EShaderUniformType::Float4DArray;   return true;
			if (TypeName == "Float4D")        rhs = ESource::EShaderUniformType::Float4D;        return true;
			if (TypeName == "Texture2D")      rhs = ESource::EShaderUniformType::Texture2D;      return true;
			if (TypeName == "Cubemap")        rhs = ESource::EShaderUniformType::Cubemap;        return true;
			if (TypeName == "Int")            rhs = ESource::EShaderUniformType::Int;            return true;
			if (TypeName == "IntArray")       rhs = ESource::EShaderUniformType::IntArray;       return true;
			return true;
		}
	};
}

namespace ESource {

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

					int ShaderFlags = ShaderProgramNode["Instancing"].IsDefined() ?
						(ShaderProgramNode["Instancing"].as<int>() > 0 ? (int)EShaderCompileFalgs::Instancing : 0) : 0;
					if (ShaderFlags & (int)EShaderCompileFalgs::Instancing) {
						CreateProgram(Name + L"#Instancing", OriginFile, "", ShaderFlags)->Load();
					}
					CreateProgram(Name, OriginFile, "", 0)->Load();
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

	RShaderPtr ShaderManager::CreateProgram(const WString & Name, const WString & Origin, const NString& Source, int CompileFlags) {
		RShaderPtr Shader = GetProgram(Name);
		if (Shader == NULL) {
			Shader = RShaderPtr(new RShader(Name, Origin, Source, CompileFlags));
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
