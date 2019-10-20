
#include "CoreMinimal.h"
#include "Core/Transform.h"
#include "Core/Application.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderScene.h"

#include "Resources/TextureManager.h"
#include "Resources/ShaderManager.h"

namespace ESource {

	RenderScene::RenderScene() {
		ModelMatrixBuffer = VertexBuffer::Create(NULL, 0, EUsageMode::UM_Dynamic);
	}

	void RenderScene::Clear() {
		RenderElementsByMaterial.clear();
		SortedMaterials.clear();
		LightCount = -1;
	}

	void RenderScene::ForwardRender() {
		for (auto & MatIt : SortedMaterials) {
			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			MatIt->SetParameters({
				{ "_ViewPosition",                { EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",            { ViewProjection }, SPFlags_IsInternal },
				{ "_ViewMatrix",                  { EyeTransform.GetGLViewMatrix() }, SPFlags_IsInternal },
				{ "_Lights[0].Position",          { Lights[0].Transformation.Position }, SPFlags_IsInternal },
				{ "_Lights[0].ProjectionMatrix",  { Lights[0].ProjectionMatrix }, SPFlags_IsInternal},
				{ "_Lights[0].ViewMatrix",        { Lights[0].Transformation.GetGLViewMatrix() }, SPFlags_IsInternal },
				{ "_Lights[0].Color",             { Lights[0].Color }, SPFlags_IsInternal | SPFlags_IsColor },
				{ "_Lights[0].Direction",         { Lights[0].Direction }, SPFlags_IsInternal },
				{ "_Lights[0].Intencity",         { Lights[0].Intensity }, SPFlags_IsInternal },
				{ "_Lights[0].ShadowMap",         { ETextureDimension::Texture2D, 
					Lights[0].CastShadow ? Lights[0].ShadowMap : TextureManager::GetInstance().GetTexture(L"WhiteTexture")}, SPFlags_IsInternal },
				{ "_Lights[0].ShadowBias",        { Lights[0].ShadowBias }, SPFlags_IsInternal },
				{ "_Lights[1].Position",          { Lights[1].Transformation.Position }, SPFlags_IsInternal },
				{ "_Lights[1].ProjectionMatrix",  { Lights[1].ProjectionMatrix }, SPFlags_IsInternal },
				{ "_Lights[1].ViewMatrix",        { Lights[1].Transformation.GetGLViewMatrix() }, SPFlags_IsInternal },
				{ "_Lights[1].Color",             { Lights[1].Color }, SPFlags_IsInternal | SPFlags_IsColor },
				{ "_Lights[1].Direction",         { Lights[1].Direction }, SPFlags_IsInternal },
				{ "_Lights[1].Intencity",         { Lights[1].Intensity }, SPFlags_IsInternal },
				{ "_Lights[1].ShadowMap",         { ETextureDimension::Texture2D,
					Lights[1].CastShadow ? Lights[1].ShadowMap : TextureManager::GetInstance().GetTexture(L"WhiteTexture")}, SPFlags_IsInternal },
				{ "_Lights[1].ShadowBias",        { Lights[1].ShadowBias }, SPFlags_IsInternal },
				{ "_GlobalTime",                  { Time::GetEpochTime<Time::Second>() }, SPFlags_IsInternal },
				{ "_BRDFLUT",                     { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"BRDFLut") }, SPFlags_IsInternal },
				{ "_EnviromentMap",               { ETextureDimension::Cubemap, EnviromentCubemap }, SPFlags_IsInternal },
				{ "_EnviromentMapLods",           { CubemapTextureMipmaps }, SPFlags_IsInternal }
				});
			MatIt->Use();

			for (auto& Element : RenderElementsByMaterial[MatIt]) {
				Element.first->Bind();
				for (auto & SubdivisionInstance : Element.second) {
					MatIt->GetShaderProgram()->GetProgram()->SetMatrix4x4Array("_ModelMatrix", std::get<Matrix4x4>(SubdivisionInstance).PointerToValue());
					Rendering::DrawIndexed(Element.first, std::get<Subdivision>(SubdivisionInstance));
				}
			}
		}
	}

	void RenderScene::DeferredRenderOpaque() {
		RShaderPtr GShader = ShaderManager::GetInstance().GetProgram(L"GBufferPass");
		if (GShader == NULL || !GShader->IsValid()) return;
		Material GMat = Material(L"GPass");
		GMat.SetShaderProgram(GShader);

		for (auto & MatIt : SortedMaterials) {
			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			MatIt->SetParameters({
				{ "_ViewPosition",                { EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",            { ViewProjection }, SPFlags_IsInternal },
				{ "_ViewMatrix",                  { EyeTransform.GetGLViewMatrix() }, SPFlags_IsInternal },
				{ "_GlobalTime",                  { Time::GetEpochTime<Time::Second>() }, SPFlags_IsInternal }
				});
			GMat.SetParameters({
				{ "_MainTexture",                 { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"WhiteTexture") }, SPFlags_IsInternal },
				{ "_NormalTexture",               { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"NormalTexture") }, SPFlags_IsInternal } 
			});
			GMat.SetParameters(MatIt->GetVariables().GetVariables());
			GMat.bWriteDepth = MatIt->bWriteDepth;
			GMat.DepthFunction = MatIt->DepthFunction;
			GMat.CullMode = MatIt->CullMode;
			GMat.FillMode = MatIt->FillMode;

			if (GMat.bWriteDepth == false) continue;

			GMat.Use();

			for (auto& Element : RenderElementsByMaterial[MatIt]) {
				Element.first->Bind();
				for (auto & SubdivisionInstance : Element.second) {
					GShader->GetProgram()->SetMatrix4x4Array("_ModelMatrix", std::get<Matrix4x4>(SubdivisionInstance).PointerToValue());
					Rendering::DrawIndexed(Element.first, std::get<Subdivision>(SubdivisionInstance));
				}
			}
		}
	}

	void RenderScene::DeferredRenderTransparent() {
		RShaderPtr GShader = ShaderManager::GetInstance().GetProgram(L"GBufferPass");
		if (GShader == NULL || !GShader->IsValid()) return;
		Material GMat = Material(L"GPass");
		GMat.SetShaderProgram(GShader);

		for (auto & MatIt : SortedMaterials) {
			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			MatIt->SetParameters({
				{ "_ViewPosition",                { EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",            { ViewProjection }, SPFlags_IsInternal },
				{ "_ViewMatrix",                  { EyeTransform.GetGLViewMatrix() }, SPFlags_IsInternal },
				{ "_GlobalTime",                  { Time::GetEpochTime<Time::Second>() }, SPFlags_IsInternal }
			});
			GMat.SetParameters({
				{ "_MainTexture",                 { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"WhiteTexture") }, SPFlags_IsInternal },
				{ "_NormalTexture",               { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"NormalTexture") }, SPFlags_IsInternal }
			});
			GMat.SetParameters(MatIt->GetVariables().GetVariables());
			GMat.bWriteDepth = MatIt->bWriteDepth;
			GMat.DepthFunction = MatIt->DepthFunction;
			GMat.CullMode = MatIt->CullMode;
			GMat.FillMode = MatIt->FillMode;

			if (GMat.bWriteDepth == false) continue;

			GMat.Use();

			for (auto& Element : RenderElementsByMaterial[MatIt]) {
				Element.first->Bind();
				for (auto & SubdivisionInstance : Element.second) {
					GShader->GetProgram()->SetMatrix4x4Array("_ModelMatrix", std::get<Matrix4x4>(SubdivisionInstance).PointerToValue());
					Rendering::DrawIndexed(Element.first, std::get<Subdivision>(SubdivisionInstance));
				}
			}
		}
	}

	void RenderScene::RenderLightMap(unsigned int LightIndex, RShaderPtr & Shader) {
		if (!Lights[LightIndex].CastShadow || Lights[LightIndex].ShadowMap == NULL || !Shader->IsValid()) return;
		Lights[LightIndex].ShadowMap->Load();
		if (Lights[LightIndex].ShadowMap->GetLoadState() != LS_Loaded) return;
		static RenderTargetPtr ShadowRenderTarget = RenderTarget::Create();
		ShadowRenderTarget->Bind();
		ShadowRenderTarget->BindDepthTexture2D((Texture2D *)Lights[LightIndex].ShadowMap->GetNativeTexture(), Lights[LightIndex].ShadowMap->GetSize());
		Rendering::SetViewport({ 0.F, 0.F, (float)Lights[LightIndex].ShadowMap->GetSize().x, (float)Lights[LightIndex].ShadowMap->GetSize().y });
		ShadowRenderTarget->Clear();

		Rendering::SetActiveDepthTest(true);
		Rendering::SetDepthFunction(DF_LessEqual);
		Rendering::SetRasterizerFillMode(FM_Solid);
		Rendering::SetCullMode(CM_None);

		RTexturePtr WhiteTexture = TextureManager::GetInstance().GetTexture(L"WhiteTexture");

		Shader->GetProgram()->Bind();
		Shader->GetProgram()->SetMatrix4x4Array("_ProjectionMatrix", Lights[LightIndex].ProjectionMatrix.PointerToValue());
		Shader->GetProgram()->SetMatrix4x4Array("_ViewMatrix", Lights[LightIndex].Transformation.GetGLViewMatrix().PointerToValue());

		for (auto & MatIt : SortedMaterials) {

			ShaderParameter * Parameter = MatIt->GetVariables().GetVariable("_MainTexture");
			if (Parameter && Parameter->Value.Texture) {
				Shader->GetProgram()->SetTexture("_MainTexture", Parameter->Value.Texture->GetNativeTexture(), 0);
			}
			else if (WhiteTexture) {
				Shader->GetProgram()->SetTexture("_MainTexture", WhiteTexture->GetNativeTexture(), 0);
			}
			else {
				Shader->GetProgram()->SetTexture("_MainTexture", NULL, 0);
			}

			for (auto& Element : RenderElementsByMaterial[MatIt]) {
				Element.first->Bind();
				for (auto & SubdivisionInstance : Element.second) {
					Shader->GetProgram()->SetMatrix4x4Array("_ModelMatrix", std::get<Matrix4x4>(SubdivisionInstance).PointerToValue());
					Rendering::DrawIndexed(Element.first, std::get<Subdivision>(SubdivisionInstance));
				}
			}
		}

		ShadowRenderTarget->CheckStatus();

		Shader->GetProgram()->Unbind();
		ShadowRenderTarget->Unbind();
	}

	void RenderScene::Submit(const MaterialPtr & Mat, const VertexArrayPtr& VertexArray, const Subdivision & MeshSubdivision, const Matrix4x4 & Matrix) {
		if (Mat == NULL || VertexArray == NULL) return;
		RenderElementsByMaterial[Mat][VertexArray].push_back(std::make_tuple(MeshSubdivision, Matrix));
		bool Inserted = false;
		for (TArray<MaterialPtr>::const_iterator MatIt = SortedMaterials.begin(); MatIt != SortedMaterials.end(); ++MatIt) {
			if (Mat == *MatIt) {
				Inserted = true;
				break;
			}
			if (*(*MatIt) > *Mat) {
				SortedMaterials.emplace(MatIt, Mat);
				Inserted = true;
				break;
			}
		}

		if (!Inserted) {
			SortedMaterials.push_back(Mat);
		}
	}

}