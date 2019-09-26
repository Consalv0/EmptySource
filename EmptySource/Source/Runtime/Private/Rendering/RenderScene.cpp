
#include "CoreMinimal.h"
#include "Core/Transform.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"
#include "Rendering/RenderScene.h"

#include "Resources/TextureManager.h"

namespace EmptySource {

	RenderScene::RenderScene() {
		ModelMatrixBuffer = VertexBuffer::Create(NULL, 0, EUsageMode::UM_Dynamic);
	}

	void RenderScene::Clear() {
		RenderElementsMaterials.clear();
		RenderElementsByMaterialID.clear();
	}

	void RenderScene::Render() {
		TArray<MaterialPtr> Materials;
		for (TDictionary<size_t, MaterialPtr>::const_iterator RMatIt = RenderElementsMaterials.begin(); RMatIt != RenderElementsMaterials.end(); ++RMatIt) {
			TArray<MaterialPtr>::const_iterator MatIt = Materials.begin();
			for (; MatIt != Materials.end(); ++MatIt) {
				if (*(*MatIt) > *(RMatIt->second))
					break;
			}
			Materials.emplace(MatIt, RMatIt->second);
		}

		TArray<MaterialPtr>::const_iterator MatIt = Materials.begin();
		for (; MatIt != Materials.end(); ++MatIt) {
			TArray<RenderElement> & RenderElements = RenderElementsByMaterialID[(*MatIt)->GetName().GetInstanceID()];

			TDictionary<VertexArrayPtr, TArray<Matrix4x4>> VertexArrayTable;
			TArray<RenderElement>::const_iterator RElementIt = RenderElements.begin();
			for (; RElementIt != RenderElements.end(); ++RElementIt) {
				VertexArrayTable.try_emplace(std::get<VertexArrayPtr>(*RElementIt), TArray<Matrix4x4>());
				VertexArrayTable[std::get<VertexArrayPtr>(*RElementIt)].push_back(std::get<Matrix4x4>(*RElementIt));
			}

			RTexturePtr EnviromentCubemap = TextureManager::GetInstance().GetTexture(L"CubemapTexture");
			float CubemapTextureMipmaps = 0.F;
			if (EnviromentCubemap)
				CubemapTextureMipmaps = (float)EnviromentCubemap->GetMipMapCount();
			(*MatIt)->SetParameters({
				{ "_ViewPosition",        { EyeTransform.Position }, SPFlags_IsInternal },
				{ "_ProjectionMatrix",    { ViewProjection }, SPFlags_IsInternal },
				{ "_ViewMatrix",          { EyeTransform.GetGLViewMatrix() }, SPFlags_IsInternal },
				{ "_Lights[0].Position",  { Lights[0].Position }, SPFlags_IsInternal },
				{ "_Lights[0].Color",     { Lights[0].Color }, SPFlags_IsInternal | SPFlags_IsColor },
				{ "_Lights[0].Intencity", { Lights[0].Intensity }, SPFlags_IsInternal },
				{ "_Lights[1].Position",  { Lights[1].Position }, SPFlags_IsInternal },
				{ "_Lights[1].Color",     { Lights[1].Color }, SPFlags_IsInternal | SPFlags_IsColor },
				{ "_Lights[1].Intencity", { Lights[1].Intensity }, SPFlags_IsInternal },
				{ "_GlobalTime",          { Time::GetEpochTime<Time::Second>() }, SPFlags_IsInternal },
				{ "_BRDFLUT",             { ETextureDimension::Texture2D, TextureManager::GetInstance().GetTexture(L"BRDFLut") }, SPFlags_IsInternal },
				{ "_EnviromentMap",       { ETextureDimension::Cubemap, EnviromentCubemap }, SPFlags_IsInternal },
				{ "_EnviromentMapLods",   { CubemapTextureMipmaps }, SPFlags_IsInternal }
				});
			(*MatIt)->Use();

			for (auto& Element : VertexArrayTable) {
				Element.first->Bind();

				float CubemapTextureMipmaps = (float)TextureManager::GetInstance().GetTexture(L"CubemapTexture")->GetMipMapCount();
				(*MatIt)->GetShaderProgram()->GetProgram()->SetAttribMatrix4x4Array("_iModelMatrix", (int)Element.second.size(), &Element.second[0], ModelMatrixBuffer);
				Rendering::DrawIndexed(Element.first, (int)Element.second.size());
			}
		}
	}

	void RenderScene::Submit(const MaterialPtr & Mat, const VertexArrayPtr& VertexArray, const Matrix4x4 & Matrix) {
		size_t InstanceID = Mat->GetName().GetInstanceID();
		if (RenderElementsMaterials.try_emplace(InstanceID, Mat).second) {
			RenderElementsByMaterialID.emplace(InstanceID, TArray<RenderElement>());
		}
		RenderElementsByMaterialID[InstanceID].push_back({ VertexArray, Matrix });
	}

}