#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"

#include "Resources/ShaderResource.h"

namespace ESource {

	class MaterialLayout {
	public:
		MaterialLayout() {}

		MaterialLayout(const TArrayInitializer<ShaderParameter> Parameters) : MaterialVariables(Parameters) { }

		const TArray<ShaderParameter>& GetVariables() const { return MaterialVariables; };

		ShaderParameter * GetVariable(const NString & Name);

		void SetParameter(const ShaderParameter& Variable);

		void AddParameter(const ShaderParameter& Property);

		TArray<ShaderParameter>::const_iterator Find(const NString & Name) const;

		TArray<ShaderParameter>::iterator Find(const NString & Name);

		void Clear() { MaterialVariables.clear(); };

		TArray<ShaderParameter>::iterator begin() { return MaterialVariables.begin(); }
		TArray<ShaderParameter>::iterator end() { return MaterialVariables.end(); }
		TArray<ShaderParameter>::const_iterator begin() const { return MaterialVariables.begin(); }
		TArray<ShaderParameter>::const_iterator end() const { return MaterialVariables.end(); }

	private:
		TArray<ShaderParameter> MaterialVariables;
	};

	class Material {
	public:
		bool bWriteDepth;
		bool bTransparent;
		bool bCastShadows;
		uint32_t RenderPriority;
		EDepthFunction DepthFunction;
		EStencilFunction StencilFunction;
		uint8_t StencilReference;
		uint8_t StencilMask;
		EStencilOperation StencilOnlyPass;
		EStencilOperation StencilOnlyFail;
		EStencilOperation StencilPassFail;
		ERasterizerFillMode FillMode;
		ECullMode CullMode;

		Material(const IName & Name);

		//* Set material shader
		void SetShaderProgram(const RShaderPtr & Value);

		//* Set instancing shader
		void SetShaderInstancingProgram(const RShaderPtr & Value);

		//* Get material shader
		RShaderPtr GetCurrentStateShaderProgram() const;

		//* Get material shader
		RShaderPtr GetShaderProgram() const;

		//* Get material shader
		RShaderPtr GetInstancingShaderProgram() const;

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
		void SetTexture2D(const NChar * UniformName, RTexturePtr Tex, const uint32_t& Position) const;

		//* Pass Cubemap array
		void SetTextureCubemap(const NChar * UniformName, RTexturePtr Tex, const uint32_t& Position) const;

		inline MaterialLayout& GetVariables() { return ParameterLayout; };

		void SetParameters(const TArray<ShaderParameter>& NewLayout);

		void AddParameters(const TArray<ShaderParameter>& NewLayout);

		//* Use shader program, asign uniform and render mode
		void Use(bool Instancing = false);

		bool operator>(const Material& Other) const;

	private:
		IName Name;
		RShaderPtr Shader;
		RShaderPtr ShaderInstancing;
		bool bHasInstancing;
		bool bIsInstancingActive;
		MaterialLayout ParameterLayout;
	};

	typedef std::shared_ptr<Material> MaterialPtr;

}