
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

#include "Utility/Hasher.h"
MAKE_HASHABLE(ESource::Subdivision, t.MaterialIndex, t.BaseVertex, t.BaseIndex, t.IndexCount);

namespace ESource {

	RenderScene::RenderScene() {
		ModelMatrixBuffer = VertexBuffer::Create(NULL, 0, EUsageMode::UM_Dynamic);
	}

	void RenderScene::Clear() {
		RenderElementsByMeshByMaterial.clear();
		SortedMaterials.clear();
		RenderElementsInstanceByMeshByMaterial.clear();
		LightCount = -1;
		CameraCount = -1;
		for (int i = 0; i < 2; ++i) {
			Lights[i].Color = 0.F;
			Lights[i].RenderMask = UINT8_MAX;
		}
		for (int i = 0; i < 2; ++i) {
			Cameras[i].RenderMask = UINT8_MAX;
		}
	}

	void RenderScene::ForwardRender(uint8_t CameraIndex) {
		for (auto & MatIt : SortedMaterials) {
			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			MatIt->SetParameters({
				{ "_ViewPosition",                { Cameras[CameraIndex].EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",            { Cameras[CameraIndex].ProjectionMatrix }, SPFlags_IsInternal },
				{ "_ViewMatrix",                  { Cameras[CameraIndex].EyeTransform.GetGLViewMatrix() }, SPFlags_IsInternal },
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

			if (RenderElementsInstanceByMeshByMaterial.find(MatIt) != RenderElementsInstanceByMeshByMaterial.end()) {
				MatIt->Use(true);
				for (auto& Element : RenderElementsInstanceByMeshByMaterial[MatIt]) {
					Element.first->GetVertexArray()->Bind();
					TDictionary<Subdivision, TArray<Matrix4x4>> Transforms;
					for (auto & ElementInstance : Element.second) {
						if (Cameras[CameraIndex].RenderMask & ElementInstance.RenderMask) {
							if (Cameras[CameraIndex].ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(ElementInstance.Transformation)) != ECullingResult::Outside) {
								Transforms[ElementInstance.MeshSubdivision].push_back(ElementInstance.Transformation);
							}
						}
					}

					for (auto& Transformation : Transforms) {
						MatIt->GetCurrentStateShaderProgram()->GetProgram()->SetAttribMatrix4x4Array("_ModelMatrix", (int)Transformation.second.size(), Transformation.second[0].PointerToValue(), ModelMatrixBuffer);
						Rendering::DrawIndexedInstanced(Element.first->GetVertexArray(), Transformation.first, (int)Transformation.second.size());
					}
				}
			}
			{
				MatIt->Use(false);
				for (auto& Element : RenderElementsByMeshByMaterial[MatIt]) {
					Element.first->GetVertexArray()->Bind();
					for (auto & ElementInstance : Element.second) {
						if (Cameras[CameraIndex].RenderMask & ElementInstance.RenderMask) {
							if (Cameras[CameraIndex].ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(ElementInstance.Transformation)) != ECullingResult::Outside) {
								MatIt->GetCurrentStateShaderProgram()->GetProgram()->SetMatrix4x4Array("_ModelMatrix", ElementInstance.Transformation.PointerToValue());
								Rendering::DrawIndexed(Element.first->GetVertexArray(), ElementInstance.MeshSubdivision);
							}
						}
					}
				}
			}
		}
	}

	void RenderScene::DeferredRenderOpaque(uint8_t CameraIndex) {
		RShaderPtr GShader = ShaderManager::GetInstance().GetProgram(L"GBufferPass");
		RShaderPtr GShaderInstancing = ShaderManager::GetInstance().GetProgram(L"GBufferPass#Instancing");
		if (GShader == NULL || !GShader->IsValid()) return;
		Material GMat = Material(L"GPass");
		GMat.SetShaderProgram(GShader);
		GMat.SetShaderInstancingProgram(GShaderInstancing);

		for (auto & MatIt : SortedMaterials) {
			if (MatIt->bTransparent) continue;
			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			MatIt->SetParameters({
				{ "_ViewPosition",                { Cameras[CameraIndex].EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",            { Cameras[CameraIndex].ProjectionMatrix }, SPFlags_IsInternal },
				{ "_ViewMatrix",                  { Cameras[CameraIndex].EyeTransform.GetGLViewMatrix() }, SPFlags_IsInternal },
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
			GMat.StencilFunction = MatIt->StencilFunction;
			GMat.StencilReference = MatIt->StencilReference;
			GMat.StencilMask = MatIt->StencilMask;
			GMat.StencilOnlyPass =  MatIt->StencilOnlyPass;
			GMat.StencilOnlyFail =  MatIt->StencilOnlyFail;
			GMat.StencilPassFail =  MatIt->StencilPassFail;

			if (RenderElementsInstanceByMeshByMaterial.find(MatIt) != RenderElementsInstanceByMeshByMaterial.end()) {
				GMat.Use(true);
				for (auto& Element : RenderElementsInstanceByMeshByMaterial[MatIt]) {
					TDictionary<Subdivision, TArray<Matrix4x4>> Transforms;
					for (auto & ElementInstance : Element.second) {
						if (Cameras[CameraIndex].RenderMask & ElementInstance.RenderMask) {
							if (Cameras[CameraIndex].ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(ElementInstance.Transformation)) != ECullingResult::Outside) {
								Transforms[ElementInstance.MeshSubdivision].push_back(ElementInstance.Transformation);
							}
						}
					}

					Element.first->GetVertexArray()->Bind();
					for (auto& Transformation : Transforms) {
						GMat.GetCurrentStateShaderProgram()->GetProgram()->SetAttribMatrix4x4Array("_ModelMatrix", (int)Transformation.second.size(), Transformation.second[0].PointerToValue(), ModelMatrixBuffer);
						Rendering::DrawIndexedInstanced(Element.first->GetVertexArray(), Transformation.first, (int)Transformation.second.size());
					}
				}
			}
			{
				GMat.Use(false);
				for (auto& Element : RenderElementsByMeshByMaterial[MatIt]) {
					Element.first->GetVertexArray()->Bind();
					for (auto & ElementInstance : Element.second) {
						if (Cameras[CameraIndex].RenderMask & ElementInstance.RenderMask) {
							if (Cameras[CameraIndex].ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(ElementInstance.Transformation)) != ECullingResult::Outside) {
								GMat.GetCurrentStateShaderProgram()->GetProgram()->SetMatrix4x4Array("_ModelMatrix", ElementInstance.Transformation.PointerToValue());
								Rendering::DrawIndexed(Element.first->GetVertexArray(), ElementInstance.MeshSubdivision);
							}
						}
					}
				}
			}
		}
	}

	void RenderScene::DeferredRenderTransparent(uint8_t CameraIndex) {
		RShaderPtr GShader = ShaderManager::GetInstance().GetProgram(L"GBufferPass");
		RShaderPtr GShaderInstancing = ShaderManager::GetInstance().GetProgram(L"GBufferPass#Instancing");
		if (GShader == NULL || !GShader->IsValid()) return;
		Material GMat = Material(L"GPass");
		GMat.SetShaderProgram(GShader);
		GMat.SetShaderInstancingProgram(GShaderInstancing);

		for (auto & MatIt : SortedMaterials) {
			if (!MatIt->bTransparent) continue;
			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			MatIt->SetParameters({
				{ "_ViewPosition",                { Cameras[CameraIndex].EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",            { Cameras[CameraIndex].ProjectionMatrix }, SPFlags_IsInternal },
				{ "_ViewMatrix",                  { Cameras[CameraIndex].EyeTransform.GetGLViewMatrix() }, SPFlags_IsInternal },
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
			GMat.StencilFunction = MatIt->StencilFunction;
			GMat.StencilReference = MatIt->StencilReference;
			GMat.StencilMask = MatIt->StencilMask;
			GMat.StencilOnlyPass = MatIt->StencilOnlyPass;
			GMat.StencilOnlyFail = MatIt->StencilOnlyFail;
			GMat.StencilPassFail = MatIt->StencilPassFail;

			if (RenderElementsInstanceByMeshByMaterial.find(MatIt) != RenderElementsInstanceByMeshByMaterial.end()) {
				GMat.Use(true);
				for (auto& Element : RenderElementsInstanceByMeshByMaterial[MatIt]) {
					Element.first->GetVertexArray()->Bind();
					TDictionary<Subdivision, TArray<Matrix4x4>> Transforms;
					for (auto & ElementInstance : Element.second) {
						if (Cameras[CameraIndex].RenderMask & ElementInstance.RenderMask) {
							if (Cameras[CameraIndex].ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(ElementInstance.Transformation)) != ECullingResult::Outside) {
								Transforms[ElementInstance.MeshSubdivision].push_back(ElementInstance.Transformation);
							}
						}
					}

					for (auto& Transformation : Transforms) {
						GShader->GetProgram()->SetAttribMatrix4x4Array("_ModelMatrix", (int)Transformation.second.size(), Transformation.second[0].PointerToValue(), ModelMatrixBuffer);
						Rendering::DrawIndexedInstanced(Element.first->GetVertexArray(), Transformation.first, (int)Transformation.second.size());
					}
				}
			}
			{
				GMat.Use(false);
				for (auto& Element : RenderElementsByMeshByMaterial[MatIt]) {
					Element.first->GetVertexArray()->Bind();
					for (auto & ElementInstance : Element.second) {
						if (Cameras[CameraIndex].RenderMask & ElementInstance.RenderMask) {
							if (Cameras[CameraIndex].ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(ElementInstance.Transformation)) != ECullingResult::Outside) {
								GShader->GetProgram()->SetMatrix4x4Array("_ModelMatrix", ElementInstance.Transformation.PointerToValue());
								Rendering::DrawIndexed(Element.first->GetVertexArray(), ElementInstance.MeshSubdivision);
							}
						}
					}
				}
			}
		}
	}

	void RenderScene::RenderLightMap(uint32_t LightIndex, MaterialPtr & Material) {
		RenderScene::Light & SelectedLight = Lights[LightIndex];
		if (!SelectedLight.CastShadow || SelectedLight.ShadowMap == NULL || Material == NULL) return;
		SelectedLight.ShadowMap->Load();
		if (SelectedLight.ShadowMap->GetLoadState() != LS_Loaded) return;

		static RenderTargetPtr ShadowRenderTarget = RenderTarget::Create();
		ShadowRenderTarget->Bind();
		ShadowRenderTarget->BindDepthTexture2D((Texture2D *)SelectedLight.ShadowMap->GetTexture(), PF_ShadowDepth, SelectedLight.ShadowMap->GetSize());
		Rendering::SetViewport({ 0, 0, SelectedLight.ShadowMap->GetSize().X, SelectedLight.ShadowMap->GetSize().Y });
		ShadowRenderTarget->Clear();

		Rendering::SetDepthWritting(true);
		Rendering::SetDepthFunction(DF_LessEqual);
		Rendering::SetRasterizerFillMode(FM_Solid);
		Rendering::SetStencilFunction(SF_Always, 0, 255);
		Rendering::SetStencilOperation(SO_Keep, SO_Keep, SO_Keep);
		Rendering::SetCullMode(CM_None);

		RTexturePtr WhiteTexture = TextureManager::GetInstance().GetTexture(L"WhiteTexture");


		auto & Shader = Material->GetShaderProgram();
		auto & InstancingShader = Material->GetInstancingShaderProgram();

		for (auto & MatIt : SortedMaterials) {
			if (!MatIt->bCastShadows) continue;

			if (RenderElementsInstanceByMeshByMaterial.find(MatIt) != RenderElementsInstanceByMeshByMaterial.end()) {
				InstancingShader->GetProgram()->Bind();
				InstancingShader->GetProgram()->SetMatrix4x4Array("_ProjectionMatrix", SelectedLight.ProjectionMatrix.PointerToValue());
				InstancingShader->GetProgram()->SetMatrix4x4Array("_ViewMatrix", SelectedLight.Transformation.GetGLViewMatrix().PointerToValue());
				ShaderParameter * Parameter = MatIt->GetVariables().GetVariable("_MainTexture");
				if (Parameter && Parameter->Value.Texture) {
					InstancingShader->GetProgram()->SetTexture("_MainTexture", Parameter->Value.Texture->GetTexture(), 0);
				}
				else if (WhiteTexture) {
					InstancingShader->GetProgram()->SetTexture("_MainTexture", WhiteTexture->GetTexture(), 0);
				}
				else {
					InstancingShader->GetProgram()->SetTexture("_MainTexture", NULL, 0);
				}

				for (auto& Element : RenderElementsInstanceByMeshByMaterial[MatIt]) {
					Element.first->GetVertexArray()->Bind();
					TDictionary<Subdivision, TArray<Matrix4x4>> Transforms;
					for (auto & ElementInstance : Element.second) {
						if (Lights[LightIndex].RenderMask & ElementInstance.RenderMask) {
							if (Lights[LightIndex].ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(ElementInstance.Transformation)) != ECullingResult::Outside) {
								Transforms[ElementInstance.MeshSubdivision].push_back(ElementInstance.Transformation);
							}
						}
					}

					for (auto& Transformation : Transforms) {
						InstancingShader->GetProgram()->SetAttribMatrix4x4Array("_ModelMatrix", (int)Transformation.second.size(), Transformation.second[0].PointerToValue(), ModelMatrixBuffer);
						Rendering::DrawIndexedInstanced(Element.first->GetVertexArray(), Transformation.first, (int)Transformation.second.size());
					}
				}
			}
			{
				Shader->GetProgram()->Bind();
				Shader->GetProgram()->SetMatrix4x4Array("_ProjectionMatrix", SelectedLight.ProjectionMatrix.PointerToValue());
				Shader->GetProgram()->SetMatrix4x4Array("_ViewMatrix", SelectedLight.Transformation.GetGLViewMatrix().PointerToValue());
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

				for (auto& Element : RenderElementsByMeshByMaterial[MatIt]) {
					Element.first->GetVertexArray()->Bind();
					for (auto & ElementInstance : Element.second) {
						if (Lights[LightIndex].RenderMask & ElementInstance.RenderMask) {
							if (Lights[LightIndex].ViewFrustrum.CheckAABox(Element.first->GetVertexData().Bounding.Transform(ElementInstance.Transformation)) != ECullingResult::Outside) {
								Shader->GetProgram()->SetMatrix4x4Array("_ModelMatrix", ElementInstance.Transformation.PointerToValue());
								Rendering::DrawIndexed(Element.first->GetVertexArray(), ElementInstance.MeshSubdivision);
							}
						}
					}
				}
			}
		}

		ShadowRenderTarget->CheckStatus();

		Shader->GetProgram()->Unbind();
		ShadowRenderTarget->Unbind();
	}

	void RenderScene::SubmitPointLight(const Transform & Transformation, const Vector3 & Color, const float & Intensity, const RTexturePtr & Texture, const float & Bias, uint8_t RenderMask) {
		if (LightCount >= 2) return; 
		LightCount++;
		Lights[LightCount].Transformation = Transformation;
		Lights[LightCount].Color = Color;
		Lights[LightCount].Direction = 0.F;
		Lights[LightCount].Intensity = Intensity;
		Lights[LightCount].ShadowMap = Texture;
		Lights[LightCount].ShadowBias = Bias;
		Lights[LightCount].CastShadow = Texture && Texture->IsValid();
		// Lights[LightCount].ViewFrustrum = Frustrum::FromProjectionViewMatrix(ProjectionMatrix * EyeTransform.GetGLViewMatrix());
		Lights[LightCount].RenderMask = RenderMask;
	}

	void RenderScene::SubmitSpotLight(const Transform & Transformation, const Vector3 & Color, const Vector3 & Direction, const float & Intensity, const Matrix4x4 & Projection, const RTexturePtr & Texture, const float & Bias, uint8_t RenderMask) {
		if (LightCount >= 2) return;
		LightCount++;
		Lights[LightCount].Transformation = Transformation;
		Lights[LightCount].Color = Color;
		Lights[LightCount].Direction = Direction;
		Lights[LightCount].Intensity = Intensity;
		Lights[LightCount].ProjectionMatrix = Projection;
		Lights[LightCount].ShadowMap = Texture;
		Lights[LightCount].ShadowBias = Bias;
		Lights[LightCount].CastShadow = Texture && Texture->IsValid();
		Lights[LightCount].ViewFrustrum = Frustrum::FromProjectionViewMatrix(Projection * Transformation.GetGLViewMatrix());
		Lights[LightCount].RenderMask = RenderMask;
	}

	void RenderScene::AddCamera(const Transform & EyeTransform, const Matrix4x4 & Projection, uint8_t RenderMask) {
		if (CameraCount >= 2) return;
		CameraCount++;
		Cameras[CameraCount].EyeTransform = EyeTransform;
		Cameras[CameraCount].ProjectionMatrix = Projection;
		Cameras[CameraCount].ViewFrustrum = Frustrum::FromProjectionViewMatrix(Projection * EyeTransform.GetGLViewMatrix());
		Cameras[CameraCount].RenderMask = RenderMask;
	}

	void RenderScene::Submit(const MaterialPtr & Mat, const RMeshPtr& MeshPtr, const Subdivision & MeshSubdivision, const Matrix4x4 & Matrix, uint8_t RenderingMask) {
		if (Mat == NULL || MeshPtr == NULL) return;
		RenderElementsByMeshByMaterial[Mat][MeshPtr].push_back({ MeshSubdivision, Matrix, RenderingMask });
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

	void RenderScene::SubmitInstance(const MaterialPtr & Mat, const RMeshPtr & MeshPtr, const Subdivision & MeshSubdivision, const Matrix4x4 & Matrix, uint8_t RenderingMask) {
		if (Mat == NULL || MeshPtr == NULL) return;
		RenderElementsInstanceByMeshByMaterial[Mat][MeshPtr].push_back({ MeshSubdivision, Matrix, RenderingMask });
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