
#include "CoreMinimal.h"
#include "Utility/TextFormatting.h"

#include "Math/MathUtility.h"
#include "Math/Vector4.h"
#include "Math/Quaternion.h"
#include "Math/Matrix4x4.h"

#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Rendering/Rendering.h"

namespace EmptySource {

	Material::Material(const WString & Name) : Name(Name) {
		MaterialShader = NULL;
		RenderPriority = 1000;
		bUseDepthTest = true;
		DepthFunction = DF_LessEqual;
		FillMode = FM_Solid;
		CullMode = CM_CounterClockWise;
	}

	void Material::SetShaderProgram(ShaderPtr Value) {
		VariableLayout.Clear();
		if (Value != NULL && Value->IsValid()) {
			MaterialShader.swap(Value);
			SetProperties(MaterialShader->GetProperties());
		} else {
			LOG_CORE_ERROR(L"The Shader Program '{}' is not a valid program", Value != NULL ? Value->GetName().c_str() : L"NULL");
			MaterialShader.reset();
		}
	}

	ShaderPtr Material::GetShaderProgram() const {
		return MaterialShader;
	}

	WString Material::GetName() const {
		return Name;
	}

	void Material::SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void * Data, const VertexBufferPtr & Buffer) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetAttribMatrix4x4Array(AttributeName, Count, Data, Buffer);
	}

	void Material::SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetMatrix4x4Array(UniformName, Data, Count);
	}

	void Material::SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetFloat1Array(UniformName, Data, Count);
	}

	void Material::SetInt1Array(const NChar * UniformName, const int * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetInt1Array(UniformName, Data, Count);
	}

	void Material::SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetFloat2Array(UniformName, Data, Count);
	}

	void Material::SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetFloat3Array(UniformName, Data, Count);
	}

	void Material::SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetFloat4Array(UniformName, Data, Count);
	}

	void Material::SetTexture2D(const NChar * UniformName, TexturePtr Text, const unsigned int & Position) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetTexture2D(UniformName, Text, Position);
	}

	void Material::SetTextureCubemap(const NChar * UniformName, TexturePtr Text, const unsigned int & Position) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		Program->SetTexture2D(UniformName, Text, Position);
	}

	void Material::SetVariables(const TArray<ShaderProperty>& NewLayout) {
		if (GetShaderProgram() == NULL) return;
		for (auto& Layout : NewLayout) {
			if (GetShaderProgram()->GetUniformLocation(Layout.Name.c_str()) != -1) {
				VariableLayout.SetVariable(Layout);
			}
			else {
				LOG_CORE_WARN("Setting variable to uniform location not present in {0} : {1}",
					Text::WideToNarrow(GetShaderProgram()->GetName()).c_str(), Layout.Name.c_str());
			}
		}
	}

	void Material::SetProperties(const TArray<ShaderProperty>& NewLayout) {
		if (GetShaderProgram() == NULL) return;
		for (auto& Layout : NewLayout) {
			if (GetShaderProgram()->GetUniformLocation(Layout.Name.c_str()) != -1) {
				VariableLayout.AddVariable(Layout);
			}
			else {
				LOG_CORE_WARN("Setting variable to uniform location not present in {0} : {1}",
					Text::WideToNarrow(GetShaderProgram()->GetName()).c_str(), Layout.Name.c_str());
			}
		}
	}

	void Material::Use() const {
		
		Rendering::SetActiveDepthTest(bUseDepthTest);
		Rendering::SetDepthFunction(DepthFunction);
		Rendering::SetRasterizerFillMode(FillMode);
		Rendering::SetCullMode(CullMode);

		if (MaterialShader && MaterialShader->IsValid()) {
			MaterialShader->Bind();
			unsigned int i = 0;
			for (auto& Uniform : VariableLayout) {
				switch (Uniform.Value.GetType()) {
				case EmptySource::EShaderPropertyType::Matrix4x4Array:
					SetMatrix4x4Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.Matrix4x4Array.size()); break;
				case EmptySource::EShaderPropertyType::Matrix4x4:
					SetMatrix4x4Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderPropertyType::FloatArray:
					SetFloat1Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.FloatArray.size()); break;
				case EmptySource::EShaderPropertyType::Float:
					SetFloat1Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderPropertyType::Float2DArray:
					SetFloat2Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.Float2DArray.size()); break;
				case EmptySource::EShaderPropertyType::Float2D:
					SetFloat2Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderPropertyType::Float3DArray:
					SetFloat3Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.Float3DArray.size()); break;
				case EmptySource::EShaderPropertyType::Float3D:
					SetFloat3Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderPropertyType::Float4DArray:
					SetFloat3Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), (int)Uniform.Value.Float3DArray.size()); break;
				case EmptySource::EShaderPropertyType::Float4D:
					SetFloat4Array(Uniform.Name.c_str(), (float *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderPropertyType::IntArray:
					SetInt1Array(Uniform.Name.c_str(), (int *)Uniform.Value.PointerToValue(), (int)Uniform.Value.IntArray.size()); break;
				case EmptySource::EShaderPropertyType::Int:
					SetInt1Array(Uniform.Name.c_str(), (int *)Uniform.Value.PointerToValue(), 1); break;
				case EmptySource::EShaderPropertyType::Texture2D:
					SetTexture2D(Uniform.Name.c_str(), Uniform.Value.Texture, i); i++; break;
				case EmptySource::EShaderPropertyType::Cubemap:
					SetTextureCubemap(Uniform.Name.c_str(), Uniform.Value.Texture, i); i++; break;
				case EmptySource::EShaderPropertyType::None:
				default:
					break;
				}
			}
		}
	}

	void MaterialLayout::SetVariable(const ShaderProperty & Variable) {
		auto Iterator = Find(Variable.Name);
		if (Iterator == end())
			MaterialVariables.push_back(Variable);
		else
			(*Iterator) = Variable.Value;
	}

	void MaterialLayout::AddVariable(const ShaderProperty & Property) {
		auto Iterator = Find(Property.Name);
		if (Iterator == end())
			MaterialVariables.push_back(Property);
	}

	TArray<ShaderProperty>::const_iterator MaterialLayout::Find(const NString & Name) const {
		return std::find_if(begin(), end(), [&Name](const ShaderProperty& Var) { return Var.Name == Name; });
	}

	TArray<ShaderProperty>::iterator MaterialLayout::Find(const NString & Name) {
		return std::find_if(begin(), end(), [&Name](const ShaderProperty& Var) { return Var.Name == Name; });
	}

}