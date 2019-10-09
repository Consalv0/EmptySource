
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/Window.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/RenderStage.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"

#include "Resources/ShaderManager.h"
#include "Resources/TextureManager.h"
#include "Rendering/MeshPrimitives.h"

#include "Utility/TextFormattingMath.h"

namespace ESource {
	
	RenderStage::RenderStage(const IName & Name, RenderPipeline * Pipeline) 
		: Name(Name), Scene(), Pipeline(Pipeline) {
	}

	void RenderStage::SubmitVertexArray(const VertexArrayPtr & VertexArray, const MaterialPtr & Mat, const Matrix4x4 & Matrix) {
		if (Mat->GetShaderProgram() == NULL || Mat->GetShaderProgram()->GetLoadState() != LS_Loaded) return;
		Scene.Submit(Mat, VertexArray, Matrix);
	}

	void RenderStage::SubmitPointLight(const Transform & Transformation, const Vector3 & Color, const float & Intensity) {
		Scene.LightCount++;
		Scene.Lights[Scene.LightCount].Transformation = Transformation;
		Scene.Lights[Scene.LightCount].Color = Color;
		Scene.Lights[Scene.LightCount].Direction = 0.F;
		Scene.Lights[Scene.LightCount].Intensity = Intensity;
		Scene.Lights[Scene.LightCount].CastShadow = false;
		Scene.Lights[Scene.LightCount].ShadowMap = NULL;
	}

	void RenderStage::SubmitSpotLight(const Transform & Transformation, const Vector3 & Color, const Vector3& Direction, const float & Intensity, const Matrix4x4 & Projection) {
		Scene.LightCount++;
		Scene.Lights[Scene.LightCount].Transformation = Transformation;
		Scene.Lights[Scene.LightCount].Color = Color;
		Scene.Lights[Scene.LightCount].Direction = Direction;
		Scene.Lights[Scene.LightCount].Intensity = Intensity;
		Scene.Lights[Scene.LightCount].ProjectionMatrix = Projection;
		Scene.Lights[Scene.LightCount].ShadowMap = NULL;
		Scene.Lights[Scene.LightCount].CastShadow = false;
	}

	void RenderStage::SubmitSpotShadowMap(const RTexturePtr & Texture, const float & Bias) {
		if (Scene.LightCount < 0) return;
		Scene.Lights[Scene.LightCount].ShadowMap = Texture;
		Scene.Lights[Scene.LightCount].ShadowBias = Bias;
		Scene.Lights[Scene.LightCount].CastShadow = true;
	}

	void RenderStage::SetEyeTransform(const Transform & EyeTransform) {
		Scene.EyeTransform = EyeTransform;
	}

	void RenderStage::SetProjectionMatrix(const Matrix4x4 & Projection) {
		Scene.ViewProjection = Projection;
	}

	void RenderStage::SetRenderTarget(const RenderTargetPtr & InTarget) {
		Target = InTarget;
	}

	void RenderStage::SetGeometryBuffer(const RenderTargetPtr & InTarget) {
		GeometryBuffer = InTarget;
	}

	void RenderStage::End() {
		if (Target == NULL) return;
		Rendering::SetAlphaBlending(BF_None, BF_None);
		Scene.RenderLightMap(0, ShaderManager::GetInstance().GetProgram(L"DepthTestShader"));
		Scene.RenderLightMap(1, ShaderManager::GetInstance().GetProgram(L"DepthTestShader")); 
		Rendering::SetViewport({ 0.F, 0.F, (float)Target->GetSize().x, (float)Target->GetSize().y });
		Rendering::SetAlphaBlending(BF_SrcAlpha, BF_OneMinusSrcAlpha);
		Target->Bind();
		Target->Clear();
		Scene.ForwardRender();
		Target->Unbind();
		Rendering::SetAlphaBlending(BF_None, BF_None);
		GeometryBuffer->Bind();
		GeometryBuffer->Clear();
		Scene.DeferredRender();
		GeometryBuffer->Unbind();
		Rendering::Flush();

		RShaderPtr BloomThresholdShader = ShaderManager::GetInstance().GetProgram(L"PostProcessingBloomThreshold");
		RShaderPtr BloomShader = ShaderManager::GetInstance().GetProgram(L"PostProcessingBloom");
		RShaderPtr RenderShader = ShaderManager::GetInstance().GetProgram(L"RenderScreenTexture");
		RTexturePtr BloomThresholdTexture = TextureManager::GetInstance().CreateTexture2D(L"PPBloomThreshold", L"", PF_RGB16F, FM_MinMagLinear, SAM_Clamp, 0); 
		if (BloomThresholdTexture) {
			if (BloomThresholdTexture->GetSize() != Target->GetSize() / 2) {
				BloomThresholdTexture->Unload();
				BloomThresholdTexture->SetSize(Target->GetSize() / 2);
				BloomThresholdTexture->Load();
			}
		}

		Rendering::SetViewport({ 0.F, 0.F, (float)(Application::GetInstance()->GetWindow().GetWidth() / 2), (float)(Application::GetInstance()->GetWindow().GetHeight() / 2) });

		static RenderTargetPtr BloomThresholdTarget = RenderTarget::Create();
		BloomThresholdTarget->BindTexture2D((Texture2D *)BloomThresholdTexture->GetNativeTexture(), BloomThresholdTexture->GetSize());
		BloomThresholdTarget->Bind();
		BloomThresholdTarget->Clear();
		if (BloomThresholdShader && BloomThresholdShader->IsValid()) {
			BloomThresholdShader->GetProgram()->Bind();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);
			BloomThresholdShader->GetProgram()->SetTexture("_MainTexture", Target->GetBindedTexture(0), 0);
			const float Treshold = 1.0F;
			BloomThresholdShader->GetProgram()->SetFloat1Array("_Threshold", &Treshold);
			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			BloomThresholdShader->GetProgram()->Unbind();
		}
		BloomThresholdTarget->Unbind();
		Rendering::Flush();
		BloomThresholdTexture->DeleteMipMaps();
		BloomThresholdTexture->GenerateMipMaps();

		RTexturePtr BloomHorizontalTexture = TextureManager::GetInstance().CreateTexture2D(L"PPBloomHorizontalPass", L"", PF_RGB16F, FM_MinMagLinear, SAM_Clamp, 0);
		if (BloomHorizontalTexture) {
			if (BloomHorizontalTexture->GetSize() != Target->GetSize() / 2) {
				BloomHorizontalTexture->Unload();
				BloomHorizontalTexture->SetSize(Target->GetSize() / 2);
				BloomHorizontalTexture->Load();
			}
		}
		RTexturePtr BloomTexture = TextureManager::GetInstance().CreateTexture2D(L"PPBloom", L"", PF_RGB16F, FM_MinMagNearest, SAM_Clamp, 0);
		if (BloomTexture) {
			if (BloomTexture->GetSize() != Target->GetSize() / 2) {
				BloomTexture->Unload();
				BloomTexture->SetSize(Target->GetSize() / 2);
				BloomTexture->Load();
			}
		}

		if (BloomShader && BloomShader->IsValid()) {
			BloomShader->GetProgram()->Bind();
		
			Vector2 Radius(10.0F, 10.0F);
			Vector2 HorizontalDirection(1.0F, 0.0F);
			Vector2 VerticalDirection(0.0F, 1.0F);
		
			static RenderTargetPtr BloomHorizontalBlurTarget = RenderTarget::Create();
			BloomHorizontalBlurTarget->BindTexture2D((Texture2D *)BloomHorizontalTexture->GetNativeTexture(), BloomHorizontalTexture->GetSize());
			BloomHorizontalBlurTarget->Bind();
			BloomHorizontalBlurTarget->Clear();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);
			BloomShader->GetProgram()->SetTexture("_MainTexture", BloomThresholdTexture->GetNativeTexture(), 0);
			BloomShader->GetProgram()->SetFloat1Array("_Radius", &Radius[0], 1);
			BloomShader->GetProgram()->SetFloat2Array("_Direction", HorizontalDirection.PointerToValue(), 1);
			BloomShader->GetProgram()->SetFloat2Array("_Resolution", Vector2(BloomThresholdTexture->GetSize()).PointerToValue(), 1);
			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			BloomHorizontalBlurTarget->Unbind();
			Rendering::Flush();
			BloomHorizontalTexture->DeleteMipMaps();
			BloomHorizontalTexture->GenerateMipMaps();
		
			static RenderTargetPtr BloomBlurTarget = RenderTarget::Create();
			BloomBlurTarget->BindTexture2D((Texture2D *)BloomTexture->GetNativeTexture(), BloomTexture->GetSize());
			BloomBlurTarget->Bind();
			BloomBlurTarget->Clear();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);
			BloomShader->GetProgram()->SetTexture("_MainTexture", BloomHorizontalTexture->GetNativeTexture(), 0);
			BloomShader->GetProgram()->SetFloat1Array("_Radius", &Radius[1], 1);
			BloomShader->GetProgram()->SetFloat2Array("_Direction", VerticalDirection.PointerToValue(), 1);
			BloomShader->GetProgram()->SetFloat2Array("_Resolution", Vector2(BloomHorizontalTexture->GetSize()).PointerToValue(), 1);
			
			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			BloomBlurTarget->Unbind();
			BloomShader->GetProgram()->Unbind();
			Rendering::Flush();
		}
		Rendering::Flush();

		RShaderPtr SSAOShader = ShaderManager::GetInstance().GetProgram(L"PostProcessingSSAO");
		RShaderPtr SSAOShaderBlur = ShaderManager::GetInstance().GetProgram(L"PostProcessingSSAOBlur");

		RTexturePtr SSAOTexture = TextureManager::GetInstance().CreateTexture2D(L"PPSSAO", L"", PF_R8, FM_MinMagNearest, SAM_Repeat);
		if (SSAOTexture) {
			if (SSAOTexture->GetSize() != Target->GetSize()) {
				SSAOTexture->Unload();
				SSAOTexture->SetSize(Target->GetSize());
				SSAOTexture->Load();
			}
		}

		RTexturePtr SSAOBlurTexture = TextureManager::GetInstance().CreateTexture2D(L"PPSSAOBlur", L"", PF_R8, FM_MinMagNearest, SAM_Repeat);
		if (SSAOBlurTexture) {
			if (SSAOBlurTexture->GetSize() != Target->GetSize()) {
				SSAOBlurTexture->Unload();
				SSAOBlurTexture->SetSize(Target->GetSize());
				SSAOBlurTexture->Load();
			}
		}

		static RenderTargetPtr SSAOTarget = RenderTarget::Create();
		SSAOTarget->BindTexture2D((Texture2D *)SSAOTexture->GetNativeTexture(), SSAOTexture->GetSize());
		SSAOTarget->Bind();
		SSAOTarget->Clear();
		if (SSAOShader && SSAOShader->IsValid()) {
			SSAOShader->GetProgram()->Bind();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);
			// Send kernel + rotation 
			SSAOShader->GetProgram()->SetFloat3Array("_Samples", (float *)&Application::GetInstance()->GetRenderPipeline().SSAOKernel[0], 64);
			SSAOShader->GetProgram()->SetMatrix4x4Array("_ProjectionMatrix", Scene.ViewProjection.PointerToValue());
			SSAOShader->GetProgram()->SetFloat2Array("_NoiseScale", (Application::GetInstance()->GetWindow().GetSize().FloatVector2() / 4.0F).PointerToValue());
			SSAOShader->GetProgram()->SetTexture("_GPosition", TextureManager::GetInstance().GetTexture(L"GPosition")->GetNativeTexture(), 0);
			SSAOShader->GetProgram()->SetTexture("_GNormal", TextureManager::GetInstance().GetTexture(L"GNormal")->GetNativeTexture(), 1);
			SSAOShader->GetProgram()->SetTexture("_NoiseTexture", TextureManager::GetInstance().GetTexture(L"SSAONoise")->GetNativeTexture(), 2);
			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
		}
		SSAOTarget->Unbind();
		Rendering::Flush();

		static RenderTargetPtr SSAOBlurTarget = RenderTarget::Create();
		SSAOBlurTarget->BindTexture2D((Texture2D *)SSAOBlurTexture->GetNativeTexture(), SSAOBlurTexture->GetSize());
		SSAOBlurTarget->Bind();
		SSAOBlurTarget->Clear();
		if (SSAOShaderBlur && SSAOShaderBlur->IsValid()) {
			SSAOShaderBlur->GetProgram()->Bind();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);
			SSAOShaderBlur->GetProgram()->SetTexture("_SSAO", SSAOTexture->GetNativeTexture(), 0);
			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
		}
		SSAOBlurTarget->Unbind();
		Rendering::Flush();
		Rendering::SetAlphaBlending(BF_SrcAlpha, BF_OneMinusSrcAlpha);

		Rendering::SetViewport({ 0.F, 0.F, (float)Application::GetInstance()->GetWindow().GetWidth(), (float)Application::GetInstance()->GetWindow().GetHeight() });

		if (RenderShader && RenderShader->IsValid()) {
			RenderShader->GetProgram()->Bind();
			Rendering::SetActiveDepthTest(false);
			Rendering::SetDepthFunction(DF_Always);
			Rendering::SetRasterizerFillMode(FM_Solid);
			Rendering::SetCullMode(CM_None);

			RenderShader->GetProgram()->SetTexture("_MainTexture", Target->GetBindedTexture(0), 0);
			RenderShader->GetProgram()->SetTexture("_BloomTexture", BloomTexture->GetNativeTexture(), 1);
			RenderShader->GetProgram()->SetTexture("_AOTexture", SSAOBlurTexture->GetNativeTexture(), 2);
			RenderShader->GetProgram()->SetFloat1Array("_Exposure", &Pipeline->Exposure, 1);
			RenderShader->GetProgram()->SetFloat1Array("_Gamma", &Pipeline->Gamma, 1);

			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			RenderShader->GetProgram()->Unbind();
		}
	}

	void RenderStage::Begin() {
		Scene.Clear();
	}

}