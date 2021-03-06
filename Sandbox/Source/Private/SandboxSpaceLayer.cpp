
#include "CoreMinimal.h"
#include "Core/EmptySource.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/RenderStage.h"
#include "Rendering/Rendering.h"
#include "Resources/MaterialManager.h"
#include "Resources/ModelManager.h"

#include "Components/ComponentRenderable.h"
#include "Components/ComponentCamera.h"
#include "Components/ComponentLight.h"
#include "Components/ComponentAnimable.h"

#include "../Public/SandboxSpaceLayer.h"
#include "../Public/CameraMovement.h"
#include "../External/IMGUI/imgui.h"

using namespace ESource;

void RenderGameObjectRecursive(GGameObject *& GameObject, TArray<NString> &NarrowMaterialNameList,
	TArray<IName> &MaterialNameList, TArray<NString> &NarrowMeshNameList, TArray<IName> &MeshNameList, SandboxSpaceLayer * AppLayer)
{
	bool TreeNode = ImGui::TreeNode(GameObject->GetName().GetNarrowInstanceName().c_str());
	if (ImGui::BeginPopupContextItem(GameObject->GetName().GetNarrowInstanceName().c_str())) {
		if (ImGui::Button("Delete")) {
			AppLayer->DeleteObject(GameObject);
			ImGui::EndPopup();
			if (TreeNode) ImGui::TreePop();
			return;
		}
		ImGui::EndPopup();
	}
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
		ImGui::PushItemWidth(-1); 
		ImGui::BeginGroup();
		ImGui::PushID("##Rotation");
		if (ImGui::DragFloat3("##Rotation", &EulerFrameRotation[0], 1.F, -180, 180)) {
			GameObject->LocalTransform.Rotation = Quaternion::FromEulerAngles(EulerFrameRotation);
		} ImGui::NextColumn();
		ImGui::PopID();
		ImGui::EndGroup();

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
		
		TArray<CCamera*> Cameras;
		GameObject->GetAllComponents<CCamera>(Cameras);
		for (auto & Camera : Cameras) {
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
				ImGui::DragFloat("##Angle", &Camera->ApertureAngle, 1.0F, 0.F, 180.F, "%.3F");
				ImGui::TextUnformatted("Culling Distances");
				ImGui::SameLine(); ImGui::PushItemWidth(-1);
				ImGui::DragFloat2("##Culling", &Camera->CullingDistances[0], 0.1F, 0.001F, 99999.F, "%.5F");
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
		CLight * Light = GameObject->GetFirstComponent<CLight>();
		if (Light != NULL) {
			bool TreeNode = ImGui::TreeNode(Light->GetName().GetNarrowInstanceName().c_str());
			if (ImGui::BeginPopupContextItem("Light Edit")) {
				if (ImGui::Button("Delete")) {
					GameObject->DestroyComponent(Light);
				}
				ImGui::EndPopup();
			}
			if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
				ImGui::SetDragDropPayload("CLight", &Light, sizeof(Light));
				ImGui::Text("Moving %s", Light->GetName().GetNarrowInstanceName().c_str());
				ImGui::EndDragDropSource();
			}
			if (TreeNode) {
				ImGui::Columns(2);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
				ImGui::Separator();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Color"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::ColorEdit3("##LightColor", &Light->Color[0], ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Intensity"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragFloat("##LightIntensity", &Light->Intensity); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Cast Shadow"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::Checkbox("##LightCastShadow", &Light->bCastShadow); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Shadow Bias"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragFloat("##LightShadowBias", &Light->ShadowMapBias, 0.01F, 0.001F, 0.1F, "%.5F"); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Aperture Angle"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragFloat("##Angle", &Light->ApertureAngle, 1.0F, 0.F, 180.F, "%.3F"); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Culling Planes"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragFloat2("##Culling", &Light->CullingPlanes[0], 0.1F, 0.001F, 99999.F, "%.5F"); ImGui::NextColumn();
				ImGui::PopStyleVar();
				ImGui::Columns(1);
				ImGui::PopItemWidth();
				ImGui::TreePop();
			}
		}
		CAnimable * Animable = GameObject->GetFirstComponent<CAnimable>();
		if (Animable != NULL) {
			bool TreeNode = ImGui::TreeNode(Animable->GetName().GetNarrowInstanceName().c_str());
			if (ImGui::BeginPopupContextItem("Animable Edit")) {
				if (ImGui::Button("Delete")) {
					GameObject->DestroyComponent(Animable);
				}
				ImGui::EndPopup();
			}
			if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
				ImGui::SetDragDropPayload("CAnimable", &Animable, sizeof(Animable));
				ImGui::Text("Moving %s", Animable->GetName().GetNarrowInstanceName().c_str());
				ImGui::EndDragDropSource();
			}
			if (TreeNode) {
				ImGui::Columns(2);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
				ImGui::Separator();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Track Name"); ImGui::NextColumn();
				if (Animable->Track != NULL) {
					ImGui::PushItemWidth(-1); ImGui::TextUnformatted(Animable->Track->Name.c_str());
				} ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Animation Speed"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragScalar("##Speed", ImGuiDataType_Double, &Animable->AnimationSpeed, 0.025F); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Current Animation Time"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragScalar("##CurrentTime", ImGuiDataType_Double, &Animable->CurrentAnimationTime, 0.01F); ImGui::NextColumn();
				ImGui::PopStyleVar();
				ImGui::Columns(1);
				ImGui::PopItemWidth();
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
						if (Renderable->GetMesh()->GetName() == MeshNameList[i]) {
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
						Renderable->SetMesh(ModelManager::GetInstance().GetMesh(MeshNameList[CurrentMeshIndex]));
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

					if (Renderable->GetMesh() && Renderable->GetMesh()->IsValid() && 
						Renderable->GetMesh()->GetVertexData().MaterialsMap.find(Iterator->first) != Renderable->GetMesh()->GetVertexData().MaterialsMap.end())
					{
						ImGui::Text("%s[%d]", Renderable->GetMesh()->GetVertexData().MaterialsMap.at(Iterator->first).c_str(), Iterator->first);
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
	Renderable->SetMesh(ModelManager::GetInstance().GetMesh(L"SphereUV:pSphere1"));
	Renderable->SetMaterialAt(0, MaterialManager::GetInstance().GetMaterial(L"RenderCubemapMaterial"));
	auto LightObj0 = CreateObject<GGameObject>(L"Light", Transform({ -11.5F, 34.5F, -5.5F }, Quaternion::FromEulerAngles({66, 56.F, 180.F}), 1.F));
	auto Light0 = LightObj0->CreateComponent<CLight>();
	Light0->ApertureAngle = 60.F;
	Light0->Intensity = 1200.F;
	Light0->CullingPlanes.X = 10.F;
	Light0->bCastShadow = true;
	auto LightObj1 = CreateObject<GGameObject>(L"Light", Transform({ 2.F, 0.5F, 0.F }, Quaternion(), 1.F));
	LightObj1->CreateComponent<CLight>()->bCastShadow = true;

	MaterialManager MaterialMng = MaterialManager::GetInstance();
	ModelManager ModelMng = ModelManager::GetInstance();
	auto Sponza = CreateObject<GGameObject>(L"Sponza", Transform());
	{
		auto FirstFloorArcs = CreateObject<GGameObject>(L"FirstFloorArcs");
		FirstFloorArcs->AttachTo(Sponza);
		auto Renderable = FirstFloorArcs->CreateComponent<CRenderable>();
		Renderable->SetMesh(ModelMng.GetMesh(L"Sponza:FirstFloorArcs"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Sponza/ColumnA"));
		Renderable->SetMaterialAt(1, MaterialMng.GetMaterial(L"Sponza/Arcs"));
	}
	{
		auto FirstFloorExterior = CreateObject<GGameObject>(L"FirstFloorExterior");
		FirstFloorExterior->AttachTo(Sponza);
		auto Renderable = FirstFloorExterior->CreateComponent<CRenderable>();
		Renderable->SetMesh(ModelMng.GetMesh(L"Sponza:FirstFloorExterior"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Sponza/Ceiling"));
		Renderable->SetMaterialAt(1, MaterialMng.GetMaterial(L"Sponza/ColumnC"));
		Renderable->SetMaterialAt(2, MaterialMng.GetMaterial(L"Sponza/Details"));
		Renderable->SetMaterialAt(3, MaterialMng.GetMaterial(L"Sponza/Arcs"));
		Renderable->SetMaterialAt(4, MaterialMng.GetMaterial(L"Sponza/Floor"));
		Renderable->SetMaterialAt(5, MaterialMng.GetMaterial(L"Sponza/Bricks"));
		Renderable->SetMaterialAt(6, MaterialMng.GetMaterial(L"Sponza/Ceiling"));
	}
	{
		auto Exterior = CreateObject<GGameObject>(L"Exterior");
		Exterior->AttachTo(Sponza);
		auto Renderable = Exterior->CreateComponent<CRenderable>();
		Renderable->SetMesh(ModelMng.GetMesh(L"Sponza:Exterior"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Sponza/Bricks"));
		Renderable->SetMaterialAt(1, MaterialMng.GetMaterial(L"Sponza/Roof"));
	}
	{
		auto SecondFloorInterior = CreateObject<GGameObject>(L"SecondFloorInterior");
		SecondFloorInterior->AttachTo(Sponza);
		auto Renderable = SecondFloorInterior->CreateComponent<CRenderable>();
		Renderable->SetMesh(ModelMng.GetMesh(L"Sponza:SecondFloorInterior"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Sponza/ColumnB"));
		Renderable->SetMaterialAt(1, MaterialMng.GetMaterial(L"Sponza/ColumnC"));
		Renderable->SetMaterialAt(2, MaterialMng.GetMaterial(L"Sponza/Bricks"));
		Renderable->SetMaterialAt(3, MaterialMng.GetMaterial(L"Sponza/Arcs"));
		Renderable->SetMaterialAt(4, MaterialMng.GetMaterial(L"Sponza/Ceiling"));
	}
	{
		auto SecondFloorExterior = CreateObject<GGameObject>(L"SecondFloorExterior");
		SecondFloorExterior->AttachTo(Sponza);
		auto Renderable = SecondFloorExterior->CreateComponent<CRenderable>();
		Renderable->SetMesh(ModelMng.GetMesh(L"Sponza:SecondFloorExterior"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Sponza/Details"));
		Renderable->SetMaterialAt(1, MaterialMng.GetMaterial(L"Sponza/Ceiling"));
		Renderable->SetMaterialAt(2, MaterialMng.GetMaterial(L"Sponza/Bricks"));
	}
	{
		auto UpperFloor = CreateObject<GGameObject>(L"UpperFloor");
		UpperFloor->AttachTo(Sponza);
		auto Renderable = UpperFloor->CreateComponent<CRenderable>();
		Renderable->SetMesh(ModelMng.GetMesh(L"Sponza:UpperFloor"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Sponza/Arcs"));
		Renderable->SetMaterialAt(1, MaterialMng.GetMaterial(L"Sponza/Bricks"));
		Renderable->SetMaterialAt(2, MaterialMng.GetMaterial(L"Sponza/Ceiling"));
	}
}

void SandboxSpaceLayer::OnRender() {
	Application::GetInstance()->GetRenderPipeline().Begin();
	Super::OnRender();
	Application::GetInstance()->GetRenderPipeline().End();
}

GGameObject * ModelHierarchyToSpaceHierarchy(SpaceLayer * Space, RModel *& Model, ModelNode * Node, GGameObject * NewObject) {
	if (NewObject == NULL)
		NewObject = Space->CreateObject<GGameObject>(Text::NarrowToWide(Node->Name), Node->LocalTransform);

	if (Node->bHasMesh) {
		auto MeshRenderer = NewObject->CreateComponent<CRenderable>();
		MeshRenderer->SetMesh(Model->GetMeshes().at(Node->MeshKey));
		MeshRenderer->SetMaterialAt(0, MaterialManager::GetInstance().GetMaterial(L"Sponza/Bricks"));
	}
	for (auto & Child : Node->Children) {
		GGameObject * ChildGO = ModelHierarchyToSpaceHierarchy(Space, Model, Child, NULL);
		ChildGO->AttachTo(NewObject);
	}
	return NewObject;
}

void SandboxSpaceLayer::OnImGuiRender() {
	ImGui::Begin("Sandbox Space");

	TArray<IName> MaterialNameList = MaterialManager::GetInstance().GetResourceNames();
	TArray<NString> NarrowMaterialNameList(MaterialNameList.size());
	for (int i = 0; i < MaterialNameList.size(); ++i)
		NarrowMaterialNameList[i] = MaterialNameList[i].GetNarrowDisplayName();

	TArray<IName> MeshNameList = ModelManager::GetInstance().GetResourceMeshNames();
	TArray<NString> NarrowMeshNameList(MeshNameList.size());
	for (int i = 0; i < MeshNameList.size(); ++i)
		NarrowMeshNameList[i] = Text::WideToNarrow((MeshNameList)[i].GetDisplayName());

	static NChar Text[20];
	ImGui::InputText("##Renderer", Text, 20);
	ImGui::SameLine();
	if (ImGui::Button("Create GObject")) {
		if (strlen(Text) > 0) {
			CreateObject<GGameObject>(Text::NarrowToWide(NString(Text)), Transform(0.F, Quaternion(), 1.F));
		}
		Text[0] = '\0';
	}

	if (ImGui::BeginDragDropTarget()) {
		if (const ImGuiPayload* Payload = ImGui::AcceptDragDropPayload("ModelHierarchy")) {
			ES_ASSERT(Payload->DataSize == sizeof(RModel), "DragDropData is empty");
			RModel * PayloadModel = *(RModel**)Payload;
			LOG_CORE_WARN(L"Payload Model {}", PayloadModel->GetName().GetDisplayName());

			if (PayloadModel->GetHierarchyParentNode() != NULL) {
				GGameObject * NewGO = ModelHierarchyToSpaceHierarchy(
					this, PayloadModel, PayloadModel->GetHierarchyParentNode(), CreateObject<GGameObject>(PayloadModel->GetName().GetDisplayName())
				);

				if (PayloadModel->GetAnimations().size() > 0) {
					CAnimable * Animable = NewGO->CreateComponent<CAnimable>();
					Animable->Track = &PayloadModel->GetAnimations()[0];
				}
				// TDictionary<size_t, GGameObject *> IndexGObjectMap;
				// IndexGObjectMap.emplace(0, CreateObject<GGameObject>(PayloadModel->GetName().GetDisplayName(), PayloadModel->GetHierarchyParentNode().LocalTransform));
				// size_t CurrentNodeIndex = 0;
				// while (CurrentNodeIndex < Nodes.size()) {
				// 	const ModelNode & Node = Nodes[CurrentNodeIndex];
				// 	for (auto & ChildIndex : Node.ChildrenIndices) {
				// 		const ModelNode & ChildNode = Nodes[ChildIndex];
				// 		GGameObject * ObjectNode = CreateObject<GGameObject>(Text::NarrowToWide(ChildNode.Name), ChildNode.LocalTransform);
				// 		if (ChildNode.bHasMesh) {
				// 			auto MeshRenderer = ObjectNode->CreateComponent<CRenderable>();
				// 			MeshRenderer->SetMesh(PayloadModel->GetMeshes().at(ChildNode.MeshKey));
				// 			MeshRenderer->SetMaterialAt(0, MaterialManager::GetInstance().GetMaterial(L"Sponza/Bricks"));
				// 		}
				// 		ObjectNode->AttachTo(IndexGObjectMap.at(CurrentNodeIndex));
				// 		IndexGObjectMap.emplace(ChildIndex, ObjectNode);
				// 	}
				// 	CurrentNodeIndex++;
				// }
			}
		}
		ImGui::EndDragDropTarget();
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