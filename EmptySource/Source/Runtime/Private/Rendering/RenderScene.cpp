
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

#include "Physics/Frustrum.h"

namespace ESource {

	RenderScene::RenderScene() {
		ModelMatrixBuffer = VertexBuffer::Create(NULL, 0, EUsageMode::UM_Dynamic);
	}

	void RenderScene::Clear() {
		RenderElementsByMaterial.clear();
		SortedMaterials.clear();
		RenderElementsInstanceByMaterial.clear();
		LightCount = -1;
		for (int i = 0; i < 2; ++i) {
			Lights[i].Color = 0.F;
		}
	}

	void RenderScene::ForwardRender() {
		Frustrum ViewFrustrum = Frustrum::FromProjectionViewMatrix(ProjectionMatrix * EyeTransform.GetGLViewMatrix());

		for (auto & MatIt : SortedMaterials) {
			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			MatIt->SetParameters({
				{ "_ViewPosition",                { EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",            { ProjectionMatrix }, SPFlags_IsInternal },
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
				Element.first->GetVertexArray()->Bind();
				for (auto & SubdivisionInstance : Element.second) {
					if (ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(std::get<Matrix4x4>(SubdivisionInstance))) != ECullingResult::Outside) {
						MatIt->GetShaderProgram()->GetProgram()->SetMatrix4x4Array("_ModelMatrix", std::get<Matrix4x4>(SubdivisionInstance).PointerToValue());
						Rendering::DrawIndexed(Element.first->GetVertexArray(), std::get<Subdivision>(SubdivisionInstance));
					}
				}
			}
		}
	}

	void RenderScene::DeferredRenderOpaque() {
		RShaderPtr GShader = ShaderManager::GetInstance().GetProgram(L"GBufferPass");
		if (GShader == NULL || !GShader->IsValid()) return;
		Frustrum ViewFrustrum = Frustrum::FromProjectionViewMatrix(ProjectionMatrix * EyeTransform.GetGLViewMatrix());
		Material GMat = Material(L"GPass");
		GMat.SetShaderProgram(GShader);

		for (auto & MatIt : SortedMaterials) {
			if (MatIt->bTransparent) continue;
			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			MatIt->SetParameters({
				{ "_ViewPosition",                { EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",            { ProjectionMatrix }, SPFlags_IsInternal },
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
				Element.first->GetVertexArray()->Bind();
				for (auto & SubdivisionInstance : Element.second) {
					if (ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(std::get<Matrix4x4>(SubdivisionInstance))) != ECullingResult::Outside) {
						GShader->GetProgram()->SetMatrix4x4Array("_ModelMatrix", std::get<Matrix4x4>(SubdivisionInstance).PointerToValue());
						Rendering::DrawIndexed(Element.first->GetVertexArray(), std::get<Subdivision>(SubdivisionInstance));
					}
				}
			}
		}
	}

	void RenderScene::DeferredRenderTransparent() {
		RShaderPtr GShader = ShaderManager::GetInstance().GetProgram(L"GBufferPass");
		if (GShader == NULL || !GShader->IsValid()) return;
		Frustrum ViewFrustrum = Frustrum::FromProjectionViewMatrix(ProjectionMatrix * EyeTransform.GetGLViewMatrix());
		Material GMat = Material(L"GPass");
		GMat.SetShaderProgram(GShader);

		for (auto & MatIt : SortedMaterials) {
			if (!MatIt->bTransparent) continue;
			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			MatIt->SetParameters({
				{ "_ViewPosition",                { EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",            { ProjectionMatrix }, SPFlags_IsInternal },
				{ "_ViewMatrix",                  { EyeTransform.GetGLViewMatrix() }, SPFlags_IsInternal },
				{ "_GlobalTime",                  { Time::GetEpochTime<Time::Second>() }, SPFlags_IsInternal }
			});
			GMat.SetParameters({
				{ "_MainTexture",                 { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"WhiteTexture") }, SPFlags_IsInternal },
				{ "_NormalTexture",               { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"NormalTexture") }, SPFlags_IsInternal },
				{ "_RoughnessTexture",            { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"WhiteTexture") }, SPFlags_IsInternal },
				{ "_MetallicTexture",             { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"BlackTexture") }, SPFlags_IsInternal },
			});
			GMat.SetParameters(MatIt->GetVariables().GetVariables());
			GMat.bWriteDepth = MatIt->bWriteDepth;
			GMat.DepthFunction = MatIt->DepthFunction;
			GMat.CullMode = MatIt->CullMode;
			GMat.FillMode = MatIt->FillMode;

			if (GMat.bWriteDepth == false) continue;

			GMat.Use();

			for (auto& Element : RenderElementsByMaterial[MatIt]) {
				Element.first->GetVertexArray()->Bind();
				for (auto & SubdivisionInstance : Element.second) {
					if (ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(std::get<Matrix4x4>(SubdivisionInstance))) != ECullingResult::Outside) {
						GShader->GetProgram()->SetMatrix4x4Array("_ModelMatrix", std::get<Matrix4x4>(SubdivisionInstance).PointerToValue());
						Rendering::DrawIndexed(Element.first->GetVertexArray(), std::get<Subdivision>(SubdivisionInstance));
					}
				}
			}
		}
	}

	void RenderScene::RenderLightMap(uint32_t LightIndex, RShaderPtr & Shader) {
		RenderScene::Light & SelectedLight = Lights[LightIndex];
		if (!SelectedLight.CastShadow || SelectedLight.ShadowMap == NULL || !Shader->IsValid()) return;
		SelectedLight.ShadowMap->Load();
		if (SelectedLight.ShadowMap->GetLoadState() != LS_Loaded) return;
		Frustrum ViewFrustrum = Frustrum::FromProjectionViewMatrix(SelectedLight.ProjectionMatrix * SelectedLight.Transformation.GetGLViewMatrix());

		static RenderTargetPtr ShadowRenderTarget = RenderTarget::Create();
		ShadowRenderTarget->Bind();
		ShadowRenderTarget->BindDepthTexture2D((Texture2D *)SelectedLight.ShadowMap->GetTexture(), PF_ShadowDepth, SelectedLight.ShadowMap->GetSize());
		Rendering::SetViewport({ 0, 0, SelectedLight.ShadowMap->GetSize().X, SelectedLight.ShadowMap->GetSize().Y });
		ShadowRenderTarget->Clear();

		Rendering::SetDepthWritting(true);
		Rendering::SetDepthFunction(DF_LessEqual);
		Rendering::SetRasterizerFillMode(FM_Solid);
		Rendering::SetCullMode(CM_None);

		RTexturePtr WhiteTexture = TextureManager::GetInstance().GetTexture(L"WhiteTexture");

		Shader->GetProgram()->Bind();
		Shader->GetProgram()->SetMatrix4x4Array("_ProjectionMatrix", SelectedLight.ProjectionMatrix.PointerToValue());
		Shader->GetProgram()->SetMatrix4x4Array("_ViewMatrix", SelectedLight.Transformation.GetGLViewMatrix().PointerToValue());

		for (auto & MatIt : SortedMaterials) {
			if (!MatIt->bCastShadows) continue;
			ShaderParameter * Parameter = MatIt->GetVariables().GetVariable("_MainTexture");
			if (Parameter && Parameter->Value.Texture) {
				Shader->GetProgram()->SetTexture("_MainTexture", Parameter->Value.Texture->GetTexture(), 0);
			}
			else if (WhiteTexture) {
				Shader->GetProgram()->SetTexture("_MainTexture", WhiteTexture->GetTexture(), 0);
			}
			else {
				Shader->GetProgram()->SetTexture("_MainTexture", NULL, 0);
			}

			for (auto& Element : RenderElementsByMaterial[MatIt]) {
				Element.first->GetVertexArray()->Bind();
				for (auto & SubdivisionInstance : Element.second) {
					if (ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(std::get<Matrix4x4>(SubdivisionInstance))) != ECullingResult::Outside) {
						Shader->GetProgram()->SetMatrix4x4Array("_ModelMatrix", std::get<Matrix4x4>(SubdivisionInstance).PointerToValue());
						Rendering::DrawIndexed(Element.first->GetVertexArray(), std::get<Subdivision>(SubdivisionInstance));
					}
				}
			}
		}

		ShadowRenderTarget->CheckStatus();

		Shader->GetProgram()->Unbind();
		ShadowRenderTarget->Unbind();
	}

	void RenderScene::Submit(const MaterialPtr & Mat, const RMeshPtr& MeshPtr, const Subdivision & MeshSubdivision, const Matrix4x4 & Matrix) {
		if (Mat == NULL || MeshPtr == NULL) return;
		RenderElementsByMaterial[Mat][MeshPtr].push_back(std::make_tuple(MeshSubdivision, Matrix));
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

	void RenderScene::SubmitInstance(const MaterialPtr & Mat, const RMeshPtr & MeshPtr, const Subdivision & MeshSubdivision, const Matrix4x4 & Matrix) {
		if (Mat == NULL || MeshPtr == NULL) return;
		RenderElementsInstanceByMaterial[Mat][MeshPtr].push_back(std::make_tuple(MeshSubdivision, Matrix));
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