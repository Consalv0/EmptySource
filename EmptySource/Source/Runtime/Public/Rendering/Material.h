#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"

#include "Resources/ShaderResource.h"

namespace EmptySource {

	class MaterialLayout {
	public:
		MaterialLayout() {}

		MaterialLayout(const TArrayInitializer<ShaderProperty> Properties) : MaterialVariables(Properties) { }

		const TArray<ShaderProperty>& GetVariables() const { return MaterialVariables; };

		ShaderProperty & GetVariable(const NString & Name) const { Find(Name); };

		void SetVariable(const ShaderProperty& Variable);

		void AddVariable(const ShaderProperty& Property);

		TArray<ShaderProperty>::const_iterator Find(const NString & Name) const;

		TArray<ShaderProperty>::iterator Find(const NString & Name);

		void Clear() { MaterialVariables.clear(); };

		TArray<ShaderProperty>::iterator begin() { return MaterialVariables.begin(); }
		TArray<ShaderProperty>::iterator end() { return MaterialVariables.end(); }
		TArray<ShaderProperty>::const_iterator begin() const { return MaterialVariables.begin(); }
		TArray<ShaderProperty>::const_iterator end() const { return MaterialVariables.end(); }

	private:
		TArray<ShaderProperty> MaterialVariables;
	};

	class Material {
	public:
		bool bWriteDepth;
		unsigned int RenderPriority;
		EDepthFunction DepthFunction;
		ERasterizerFillMode FillMode;
		ECullMode CullMode;

		Material(const IName & Name);

		//* Set material shader
		void SetShaderProgram(const RShaderPtr & Value);

		//* Get material shader
		RShaderPtr GetShaderProgram() const;

		const IName & GetName() const;

		//* Pass Matrix4x4 Buffer Array
		void SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void* Data, const VertexBufferPtr& Buffer) const;

		//* Pass Matrix4x4 Array
		void SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass one float vector value array
		void SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass one int vector value array
		void SetInt1Array(const NChar * UniformName, const int * Data, const int & Count = 1) const;

		//* Pass two float vector value array
		void SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass three float vector value array
		void SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass four float vector value array
		void SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass Texture 2D array
		void SetTexture2D(const NChar * UniformName, TexturePtr Tex, const unsigned int& Position) const;

		//* Pass Cubemap array
		void SetTextureCubemap(const NChar * UniformName, TexturePtr Tex, const unsigned int& Position) const;

		inline MaterialLayout& GetVariables() { return VariableLayout; };

		void SetVariables(const TArray<ShaderProperty>& NewLayout);

		void SetProperties(const TArray<ShaderProperty>& NewLayout);

		//* Use shader program, asign uniform and render mode
		void Use() const;

		bool operator>(const Material& Other) const;

	private:
		IName Name;
		RShaderPtr Shader;
		MaterialLayout VariableLayout;
	};

	typedef std::shared_ptr<Material> MaterialPtr;

}