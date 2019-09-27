
#include "CoreMinimal.h"
#include "Core/EmptySource.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/RenderStage.h"
#include "Rendering/Rendering.h"
#include "Resources/MaterialManager.h"
#include "Resources/MeshManager.h"

#include "Components/ComponentRenderable.h"
#include "Components/ComponentCamera.h"

#include "../Public/SandboxSpaceLayer.h"
#include "../Public/CameraMovement.h"
#include "../External/IMGUI/imgui.h"

using namespace ESource;

void RenderGameObjectRecursive(GGameObject *& GameObject, TArray<NString> &NarrowMaterialNameList,
	TArray<IName> &MaterialNameList, TArray<NString> &NarrowMeshNameList, TArray<WString> &MeshNameList, SandboxSpaceLayer * AppLayer)
{
	bool TreeNode = ImGui::TreeNode(GameObject->GetName().GetNarrowInstanceName().c_str());
	if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
		ImGui::SetDragDropPayload("GGameObject", &GameObject, sizeof(GGameObject));
		ImGui::Text("Moving %s", GameObject->GetName().GetNarrowInstanceName().c_str());
		ImGui::EndDragDropSource();
	}
	if (ImGui::BeginDragDropTarget()) {
		if (const ImGuiPayload* Payload = ImGui::AcceptDragDropPayload("GGameObject")) {
			ES_ASSERT(Payload->DataSize == sizeof(GGameObject), "DragDropData is empty");
			GGameObject * PayloadGameObject = *(GGameObject**)Payload->Data;
			if (GameObject->Contains(PayloadGameObject))
				PayloadGameObject->DeatachFromParent();
			else
				PayloadGameObject->AttachTo(GameObject);
		}
		ImGui::EndDragDropTarget();
	}
	if (TreeNode) {
		static NChar Text[20];
		ImGui::InputText("##Child", Text, 20);
		ImGui::SameLine();
		if (ImGui::Button("Create Child")) {
			if (strlen(Text) > 0) {
				GGameObject * ChildGameObject = AppLayer->CreateObject<GGameObject>(Text::NarrowToWide(NString(Text)), Transform(0.F, Quaternion(), 1.F));
				ChildGameObject->AttachTo(GameObject);
			}
			Text[0] = '\0';
		}
		ImGui::Columns(2);
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
		ImGui::Separator();

		ImGui::AlignTextToFramePadding(); ImGui::Text("Position"); ImGui::NextColumn();
		ImGui::PushItemWidth(-1); ImGui::DragFloat3("##Position", &GameObject->LocalTransform.Position[0], .5F, -MathConstants::BigNumber, MathConstants::BigNumber);
		ImGui::NextColumn();

		Vector3 EulerFrameRotation = GameObject->LocalTransform.Rotation.ToEulerAngles();
		ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Rotation"); ImGui::NextColumn();
		ImGui::PushItemWidth(-1); if (ImGui::DragFloat3("##Rotation", &EulerFrameRotation[0], 1.F, -180, 180)) {
			GameObject->LocalTransform.Rotation = Quaternion::EulerAngles(EulerFrameRotation);
		} ImGui::NextColumn();

		ImGui::AlignTextToFramePadding(); ImGui::Text("Scale"); ImGui::NextColumn();
		ImGui::PushItemWidth(-1); ImGui::DragFloat3("##Scale", &GameObject->LocalTransform.Scale[0], .01F, -MathConstants::BigNumber, MathConstants::BigNumber);
		ImGui::NextColumn();

		ImGui::Separator();
		ImGui::PopStyleVar();
		ImGui::Columns(1);
		ImGui::PopItemWidth();

		TArray<GGameObject *> GameObjectChildren;
		GameObject->GetAllChildren<GGameObject>(GameObjectChildren);
		for (auto & GameObjectChild : GameObjectChildren)
			RenderGameObjectRecursive(GameObjectChild, NarrowMaterialNameList, MaterialNameList, NarrowMeshNameList, MeshNameList, AppLayer);

		CCamera * Camera = GameObject->GetFirstComponent<CCamera>();
		if (Camera != NULL) {
			bool TreeNode = ImGui::TreeNode(Camera->GetName().GetNarrowInstanceName().c_str());
			if (ImGui::BeginPopupContextItem("Camera Edit")) {
				if (ImGui::Button("Delete")) {
					GameObject->DestroyComponent(Camera);
				}
				ImGui::EndPopup();
			}
			if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
				ImGui::SetDragDropPayload("CCamera", &Camera, sizeof(Camera));
				ImGui::Text("Moving %s", Camera->GetName().GetNarrowInstanceName().c_str());
				ImGui::EndDragDropSource();
			}
			if (TreeNode) {
				ImGui::TextUnformatted("Aperture Angle");
				ImGui::SameLine(); ImGui::PushItemWidth(-1);
				ImGui::DragFloat("##Angle", &Camera->ApertureAngle, 1.0F, 0.F, 360.F, "%.2F");
				ImGui::TextUnformatted("Culling Planes");
				ImGui::SameLine(); ImGui::PushItemWidth(-1);
				ImGui::DragFloat2("##Angle", &Camera->CullingPlanes[0], 1.0F, 0.1F, 99999.F, "%.2F");
				ImGui::TreePop();
			}
		}
		CCameraMovement * CameraMovement = GameObject->GetFirstComponent<CCameraMovement>();
		if (CameraMovement != NULL) {
			bool TreeNode = ImGui::TreeNode(CameraMovement->GetName().GetNarrowInstanceName().c_str());
			if (ImGui::BeginPopupContextItem("Camera Movement Edit")) {
				if (ImGui::Button("Delete")) {
					GameObject->DestroyComponent(CameraMovement);
				}
				ImGui::EndPopup();
			}
			if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
				ImGui::SetDragDropPayload("CCameraMovement", &CameraMovement, sizeof(CameraMovement));
				ImGui::Text("Moving %s", CameraMovement->GetName().GetNarrowInstanceName().c_str());
				ImGui::EndDragDropSource();
			}
			if (TreeNode) {
				ImGui::TextUnformatted("Speed");
				ImGui::SameLine(); ImGui::PushItemWidth(-1);
				ImGui::DragFloat("##ViewSpeed", &CameraMovement->ViewSpeed, 1.0F);
				ImGui::TreePop();
			}
		}
		CRenderable * Renderable = GameObject->GetFirstComponent<CRenderable>();
		if (Renderable == NULL && ImGui::Button("Create Renderer")) {
			Renderable = GameObject->CreateComponent<CRenderable>();
		}
		if (Renderable != NULL) {
			bool TreeNode = ImGui::TreeNode(Renderable->GetName().GetNarrowInstanceName().c_str()); 
			if (ImGui::BeginPopupContextItem("Renderable Edit")) {
				if (ImGui::Button("Delete")) {
					GameObject->DestroyComponent(Renderable);
				}
				ImGui::EndPopup();
			}
			if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
				ImGui::SetDragDropPayload("CRenderable", &Renderable, sizeof(CRenderable));
				ImGui::Text("Moving %s", Renderable->GetName().GetNarrowInstanceName().c_str());
				ImGui::EndDragDropSource();
			}
			if (TreeNode) {
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
				if (ImGui::Combo(("##Mesh" + std::to_string(Renderable->GetName().GetInstanceID())).c_str(), &CurrentMeshIndex,
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

					if (Renderable->GetMesh()) {
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
				}
				ImGui::TreePop();
			}
		}
		ImGui::TreePop();
	}
}

void SandboxSpaceLayer::OnAwake() {
	Super::OnAwake();
	Application::GetInstance()->GetRenderPipeline().CreateStage<RenderStage>(L"MainStage");
	auto Camera = CreateObject<GGameObject>(L"MainCamera", Transform(0.F, Quaternion(), 1.F));
	Camera->CreateComponent<CCamera>();
	Camera->CreateComponent<CCameraMovement>();
	auto SkyBox = CreateObject<GGameObject>(L"SkyBox", Transform(0.F, Quaternion(), 1000.F));
	SkyBox->AttachTo(Camera);
	auto Renderable = SkyBox->CreateComponent<CRenderable>();
	Renderable->SetMesh(MeshManager::GetInstance().GetMesh(L"pSphere1"));
	Renderable->SetMaterialAt(0, MaterialManager::GetInstance().GetMaterial(L"Cubemap"));
}

void SandboxSpaceLayer::OnRender() {
	Application::GetInstance()->GetRenderPipeline().BeginStage(L"MainStage");
	Super::OnRender();
	Application::GetInstance()->GetRenderPipeline().EndStage();
}

void SandboxSpaceLayer::OnImGuiRender() {
	ImGui::Begin("Sandbox Space");

	TArray<IName> MaterialNameList = MaterialManager::GetInstance().GetResourceNames();
	TArray<NString> NarrowMaterialNameList(MaterialNameList.size());
	for (int i = 0; i < MaterialNameList.size(); ++i)
		NarrowMaterialNameList[i] = MaterialNameList[i].GetNarrowDisplayName();

	TArray<WString> MeshNameList = MeshManager::GetInstance().GetResourceNames();
	TArray<NString> NarrowMeshNameList(MeshNameList.size());
	for (int i = 0; i < MeshNameList.size(); ++i)
		NarrowMeshNameList[i] = Text::WideToNarrow((MeshNameList)[i]);

	static NChar Text[20];
	ImGui::InputText("##Renderer", Text, 20);
	ImGui::SameLine();
	if (ImGui::Button("Create GObject")) {
		if (strlen(Text) > 0) {
			CreateObject<GGameObject>(Text::NarrowToWide(NString(Text)), Transform(0.F, Quaternion(), 1.F));
		}
		Text[0] = '\0';
	}
	ImGui::Separator();

	TArray<GGameObject *> GameObjects;
	GetAllObjects<GGameObject>(GameObjects);
	for (auto & GameObject : GameObjects)
		if (GameObject->IsRoot()) {
			RenderGameObjectRecursive(GameObject, NarrowMaterialNameList, MaterialNameList, NarrowMeshNameList, MeshNameList, this);
		}

	ImGui::End();
}

SandboxSpaceLayer::SandboxSpaceLayer(const ESource::WString & Name, unsigned int Level) : SpaceLayer(Name, Level) {
}