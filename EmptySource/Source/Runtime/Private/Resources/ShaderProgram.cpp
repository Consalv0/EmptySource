
#include "CoreMinimal.h"
#include "Resources/ShaderManager.h"
#include "Resources/ShaderProgram.h"
#include "Resources/TextureManager.h"

#include <yaml-cpp/yaml.h>

namespace EmptySource {

	bool RShaderProgram::IsValid() {
		return LoadState == LS_Loaded && ShaderPointer->IsValid();
	}

	void RShaderProgram::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			LOG_CORE_DEBUG(L"Loading Shader {}...", Name.GetDisplayName().c_str());
			if (!Origin.empty()) {
				FileStream * ShaderFile = FileManager::GetFile(Origin);
				if (ShaderFile == NULL) {
					LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}
				if (!ShaderFile->ReadNarrowStream(&Source)) {
					ShaderFile->Close();
					LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}
			}
		}
		LoadState = LoadFromShaderSource(Source) ? LS_Loaded : LS_Unloaded;
	}

	void RShaderProgram::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		for (auto& Stage : Stages)
			delete Stage;
		delete ShaderPointer;
		ShaderPointer = NULL;
		Stages.clear();
		LoadState = LS_Unloaded;
	}

	void RShaderProgram::Reload() {
		Unload();
		Load();
	}

	void RShaderProgram::SetProperties(const TArrayInitializer<ShaderProperty> & InProperties) {
		Properties = InProperties;
	}

	void RShaderProgram::SetProperties(const TArray<ShaderProperty>& InProperties) {
		Properties = InProperties;
	}

	RShaderProgram::RShaderProgram(const IName & Name, const WString & Origin, const NString& Source)
		: ResourceHolder(Name, Origin), Properties(), Source(Source), Stages() {
	}

	bool RShaderProgram::LoadFromShaderSource(const NString & FileInfo) {
		YAML::Node BaseNode; {
			try {
				BaseNode = YAML::Load(FileInfo.c_str());
			} catch (...) {
				return false;
			}
		}

		///// TODO: This may be retrieved from the renderer in the future
		YAML::Node CodeNode = BaseNode["GLSL"];

		YAML::Node NameNode = BaseNode["Name"];
		YAML::Node ParametersNode = BaseNode["Parameters"];
		YAML::Node StagesNode = CodeNode["Stages"];

		WString ShaderName = L"ShaderProgram";
		if (NameNode.IsDefined()) ShaderName = Text::NarrowToWide(NameNode.as<NString>());

		if (CodeNode.IsDefined()) {
			if (StagesNode.IsDefined()) {
				for (YAML::iterator Iterator = StagesNode.begin(); Iterator != StagesNode.end(); ++Iterator) {
					const YAML::Node& Stage = *Iterator;
					NString Type = Stage["StageType"].as<NString>();

					if (Stage["Code"].IsDefined()) {
						Stages.push_back(ShaderStage::CreateFromText("#version 410\n" + Stage["Code"].as<NString>(), ShaderManager::StringToStageType(Type)));
					}
				}
			}
		}
		ShaderPointer = ShaderProgram::Create(Stages);

		if (ParametersNode.IsDefined()) {
			TArray<ShaderProperty> Parameters;
			for (auto & Uniform : ParametersNode) {
				NString UniformName = Uniform["Uniform"].as<NString>();
				NString TypeName = Uniform["Type"].as<NString>();
				int Flags = SPFlags_None;
				if (Uniform["IsColor"]) {
					Flags |= Uniform["IsColor"].as<bool>() ? SPFlags_IsColor : SPFlags_None;
				}
				if (TypeName == "None")           Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(), Flags);
				else if (TypeName == "Matrix4x4Array") Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<Matrix4x4>()), Flags);
				else if (TypeName == "Matrix4x4")      Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(Matrix4x4()), Flags);
				else if (TypeName == "FloatArray")     Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<float>()), Flags);
				else if (TypeName == "Float")          Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(Uniform["DefaultValue"].as<float>()), Flags);
				else if (TypeName == "Float2DArray")   Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<Vector2>()), Flags);
				else if (TypeName == "Float2D")        Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(Vector2(
					Uniform["DefaultValue"][0].as<float>(), Uniform["DefaultValue"][1].as<float>())
				), Flags);
				else if (TypeName == "Float3DArray")   Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<Vector3>()), Flags);
				else if (TypeName == "Float3D")        Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(Vector3(
					Uniform["DefaultValue"][0].as<float>(), Uniform["DefaultValue"][1].as<float>(), Uniform["DefaultValue"][2].as<float>())
				), Flags);
				else if (TypeName == "Float4DArray")   Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<Vector4>()), Flags);
				else if (TypeName == "Float4D")        Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(Vector4(
					Uniform["DefaultValue"][0].as<float>(), Uniform["DefaultValue"][1].as<float>(), Uniform["DefaultValue"][2].as<float>(),
					Uniform["DefaultValue"][3].as<float>())
				), Flags);
				else if (TypeName == "Texture2D")      Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(ETextureDimension::Texture2D,
					TextureManager::GetInstance().GetTexture(Text::NarrowToWide(Uniform["DefaultValue"].as<NString>()))
				), Flags);
				else if (TypeName == "Cubemap")        Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(ETextureDimension::Cubemap,
					NULL), Flags);
				else if (TypeName == "IntArray")       Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(TArray<int>()), Flags);
				else if (TypeName == "Int")            Parameters.emplace_back(UniformName, ShaderProperty::PropertyValue(0), Flags);
			}

			SetProperties(Parameters);
			return true;
		}

		return false;
	}

	RShaderProgram::~RShaderProgram() {
		Unload();
	}

}