
#include "CoreMinimal.h"
#include "Core/EmptySource.h"
#include "Rendering/Rendering.h"
#include "Resources/MaterialManager.h"
#include "Resources/MeshManager.h"

#include "Components/ComponentRenderable.h"

#include "../Public/SandboxSpaceLayer.h"
#include "../External/IMGUI/imgui.h"

using namespace EmptySource;

SandboxSpaceLayer::SandboxSpaceLayer(const EmptySource::WString & Name, unsigned int Level) : SpaceLayer(Name, Level) {
}

void SandboxSpaceLayer::OnAwake() {
	GGameObject * GameObject = CreateObject<GGameObject>(L"TestObject", Transform(0.F, Quaternion(), 1.F));
	GameObject->CreateComponent<CRenderable>();
	Super::OnAwake();
}

void SandboxSpaceLayer::OnImGuiRender() {
	ImGui::Begin("Sandbox Space");

	TArray<WString> MaterialNameList = MaterialManager::GetInstance().GetResourceNames();
	TArray<NString> NarrowMaterialNameList(MaterialNameList.size());
	for (int i = 0; i < MaterialNameList.size(); ++i)
		NarrowMaterialNameList[i] = Text::WideToNarrow((MaterialNameList)[i]);

	TArray<WString> MeshNameList = MeshManager::GetInstance().GetResourceNames();
	TArray<NString> NarrowMeshNameList(MeshNameList.size());
	for (int i = 0; i < MeshNameList.size(); ++i)
		NarrowMeshNameList[i] = Text::WideToNarrow((MeshNameList)[i]);

	GGameObject * GameObject = GetFirstObject<GGameObject>();
	if (GameObject != NULL)
		if (ImGui::TreeNode(Text::WideToNarrow(GameObject->GetUniqueName()).c_str())) {
			CRenderable * Renderable = GameObject->GetFirstComponent<CRenderable>();
			if (Renderable != NULL) {
				if (ImGui::TreeNode(Text::WideToNarrow(Renderable->GetUniqueName()).c_str())) {
					int CurrentMeshIndex = Renderable->GetMesh() ? 0 : -1;
					if (CurrentMeshIndex == 0) {
						for (int i = 0; i < MeshNameList.size(); i++) {
							if (Renderable->GetMesh()->GetMeshData().Name == MeshNameList[i]) {
								CurrentMeshIndex = i; break;
							}
						}
					}
					ImGui::TextUnformatted("Mesh");
					ImGui::SameLine(); ImGui::PushItemWidth(-1);
					if (ImGui::Combo(("##Mesh" + std::to_string(Renderable->GetUniqueID())).c_str(), &CurrentMeshIndex,
						[](void * Data, int indx, const char ** outText) -> bool {
						TArray<NString>* Items = (TArray<NString> *)Data;
						if (outText) *outText = (*Items)[indx].c_str();
						return true;
					}, &NarrowMeshNameList, (int)NarrowMeshNameList.size())) {
						if (CurrentMeshIndex >= 0 && CurrentMeshIndex < MeshNameList.size())
							Renderable->SetMesh(MeshManager::GetInstance().GetMesh(MeshNameList[CurrentMeshIndex]));
					}
					ImGui::TextUnformatted("Materials: ");
					for (TDictionary<int, MaterialPtr>::const_iterator Iterator = Renderable->GetMaterials().begin();
						Iterator != Renderable->GetMaterials().end(); Iterator++)
					{
						int CurrentMaterialIndex = Iterator->second ? 0 : -1;
						if (CurrentMaterialIndex == 0) {
							for (int i = 0; i < MaterialNameList.size(); i++) {
								if (Iterator->second->GetName() == MaterialNameList[i]) {
									CurrentMaterialIndex = i;
									break;
								}
							}
						}

						ImGui::Text("%s[%d]", Renderable->GetMesh()->GetMeshData().Materials.at(Iterator->first).c_str(), Iterator->first);
						ImGui::SameLine(); ImGui::PushItemWidth(-1);
						if (ImGui::Combo(("##Material" + std::to_string(Iterator->first)).c_str(), &CurrentMaterialIndex,
							[](void * Data, int indx, const char ** outText) -> bool {
							TArray<NString>* Items = (TArray<NString> *)Data;
							if (outText) *outText = (*Items)[indx].c_str();
							return true;
						}, &NarrowMaterialNameList, (int)NarrowMaterialNameList.size())) {
							if (CurrentMaterialIndex >= 0 && CurrentMaterialIndex < MaterialNameList.size())
								Renderable->SetMaterialAt(Iterator->first, MaterialManager::GetInstance().GetMaterial(MaterialNameList[CurrentMaterialIndex]));
						}
					}
					ImGui::TreePop();
				}
			}
			ImGui::TreePop();
		}

	ImGui::End();
}
