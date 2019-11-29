
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
#include "Components/ComponentPhysicBody.h"

#include "../Public/GameSpaceLayer.h"
#include "../Public/CameraMovement.h"
#include "../Public/GunComponent.h"
#include "../Public/PropComponent.h"
#include "../Public/ScenePropComponent.h"
#include "../Public/GameStateComponent.h"
#include "../Public/FollowTarget.h"
#include "../External/IMGUI/imgui.h"

void RenderGameObjectRecursive(ESource::GGameObject *& GameObject, ESource::TArray<ESource::NString> &NarrowMaterialNameList,
	ESource::TArray<ESource::IName> &MaterialNameList, ESource::TArray<ESource::NString> &NarrowMeshNameList, ESource::TArray<ESource::IName> &MeshNameList, GameSpaceLayer * AppLayer)
{
	using namespace ESource;
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
				ImGui::TextUnformatted("Rendering Mask");
				ImGui::SameLine(); ImGui::PushItemWidth(-1);
				ImGui::InputScalar("##RenderingMask", ImGuiDataType_U8, &Camera->RenderingMask);
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
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Rendering Mask"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::InputScalar("##RenderingMask", ImGuiDataType_U8, &Light->RenderingMask);
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
				ImGui::TextUnformatted("Culling Mask");
				ImGui::SameLine(); ImGui::PushItemWidth(-1);
				ImGui::InputScalar("##RenderingMask", ImGuiDataType_U8, &Renderable->RenderingMask);
				ImGui::TextUnformatted("GPU Instancing");
				ImGui::SameLine(); ImGui::PushItemWidth(-1);
				ImGui::Checkbox("##GPUInstancing", &Renderable->bGPUInstancing);
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

void GameSpaceLayer::OnAwake() {
	Super::OnAwake();

	auto MainCamera = CreateObject<ESource::GGameObject>(L"MainCamera", ESource::Transform(ESource::Point3(0.F, 1.8F, 0.F), Quaternion(), 1.F));
	auto PropCameraOffset = CreateObject<ESource::GGameObject>(L"PropCamera",
		ESource::Transform(ESource::Point3(0.F, 1.0F, 0.F), Quaternion::FromEulerAngles(Vector3(5.F, 140.F, 0.F)), 1.F));
	auto PropCamera = CreateObject<ESource::GGameObject>(L"PropCameraOffset", ESource::Transform(ESource::Point3(0.F, 0.0F, -1.8F), Quaternion(), 1.F));
	PropCamera->AttachTo(PropCameraOffset);
	auto CameraComponent1 = MainCamera->CreateComponent<ESource::CCamera>();
	auto CameraComponent2 = PropCamera->CreateComponent<ESource::CCamera>();
	CameraComponent1->RenderingMask = 1 << 0;
	CameraComponent2->RenderingMask = 1 << 1;
	auto MainCameraMovement = MainCamera->CreateComponent<CCameraMovement>();
	MainCameraMovement->InputIndex = 0;
	MainCameraMovement->DefaultHeight = 1.8F;
	auto PropCameraMovement = PropCameraOffset->CreateComponent<CCameraMovement>();
	PropCameraMovement->InputIndex = 1;
	PropCameraMovement->DefaultHeight = 0.F;
	{
		auto SkyBox = CreateObject<ESource::GGameObject>(L"SkyBox", ESource::Transform(0.F, Quaternion(), 1000.F));
		SkyBox->AttachTo(MainCamera);
		auto Renderable = SkyBox->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 0;
		Renderable->SetMesh(ESource::ModelManager::GetInstance().GetMesh(L"SphereUV:pSphere1"));
		Renderable->SetMaterialAt(0, ESource::MaterialManager::GetInstance().GetMaterial(L"RenderCubemapMaterial")); 
	}
	{
		auto SkyBox = CreateObject<ESource::GGameObject>(L"SkyBox", ESource::Transform(0.F, Quaternion(), 1000.F));
		SkyBox->AttachTo(PropCamera);
		auto Renderable = SkyBox->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 1;
		Renderable->SetMesh(ESource::ModelManager::GetInstance().GetMesh(L"SphereUV:pSphere1"));
		Renderable->SetMaterialAt(0, ESource::MaterialManager::GetInstance().GetMaterial(L"RenderCubemapMaterial"));
	}
	auto LightObj0 = CreateObject<ESource::GGameObject>(L"Light", ESource::Transform({ -11.5F, 34.5F, -5.5F }, Quaternion::FromEulerAngles({85.F, 0.F, 0.F}), 1.F));
	auto Light0 = LightObj0->CreateComponent<ESource::CLight>();
	Light0->ApertureAngle = 123.F;
	Light0->Intensity = 2200.F;
	Light0->CullingPlanes.X = 15.F;
	Light0->Color = ESource::Vector3(1.F, 0.982F, 0.9F);
	Light0->bCastShadow = true;
	Light0->SetShadowMapSize(2048);
	auto FollowLight0 = LightObj0->CreateComponent<CFollowTarget>();
	FollowLight0->Target = MainCamera;
	FollowLight0->FixedPositionAxisY = true;
	auto LightObj1 = CreateObject<ESource::GGameObject>(L"Light", ESource::Transform({ 2.F, -1.5F, 0.F }, ESource::Quaternion(), 1.F));
	auto Light1 = LightObj1->CreateComponent<ESource::CLight>();
	Light1->bCastShadow = false;
	Light1->Color = ESource::Vector3(1.F, 0.982F, 0.9F);

	auto GameStateObject = CreateObject<ESource::GGameObject>(L"GameState", ESource::Transform());
	CGameState * GameState = GameStateObject->CreateComponent<CGameState>();
	PropCameraMovement->GameStateComponent = GameState;
	MainCameraMovement->GameStateComponent = GameState;

	ESource::MaterialManager MaterialMng = ESource::MaterialManager::GetInstance();
	ESource::ModelManager ModelMng = ESource::ModelManager::GetInstance();
	{
		auto Ground = CreateObject<ESource::GGameObject>(L"Ground", ESource::Transform());
		CFollowTarget * Follow = Ground->CreateComponent<CFollowTarget>();
		Follow->Target = MainCamera;
		Follow->FixedPositionAxisY = true;
		Follow->ModuleMovement = 6.F;
		auto TileDesert = ModelMng.GetMesh(L"TileDesertSends:TileDesertSends");
		auto TileBricks = ModelMng.GetMesh(L"TileDesertSends:TileGroundBricks");
		const int GridSize = 32;
		for (int i = 0; i < GridSize; i++) {
			for (int j = 0; j < GridSize; j++) {
				auto SandTile = CreateObject<ESource::GGameObject>(L"SandTile");
				SandTile->LocalTransform.Position = (ESource::Vector3(float(i), 0.F, float(j)) * 6.F) - ESource::Vector3(float(GridSize), 0.F, float(GridSize)) * 3.F;
				SandTile->AttachTo(Ground);
				auto PhysicsBody = SandTile->CreateComponent<ESource::CPhysicBody>();
				PhysicsBody->SetMesh(TileDesert);
				auto Renderable = SandTile->CreateComponent<ESource::CRenderable>();
				Renderable->bGPUInstancing = true;
				Renderable->RenderingMask = 1 << 0 | 1 << 1;
				Renderable->SetMesh(TileDesert);
				Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Tiles/DesertSends"));
			}
		}
	}

	{
		auto EgyptianCat = CreateObject<ESource::GGameObject>(
			L"EgyptianCat", ESource::Transform(Vector3(-34.F, 0.F, 70.F), Quaternion::FromEulerAngles({18.F, -16.F, 34.F}), 5.F)
		);
		auto PhysicsBody = EgyptianCat->CreateComponent<ESource::CPhysicBody>();
		PhysicsBody->SetMesh(ModelMng.GetMesh(L"EgyptianCat:Cat_Statue_CatStatue"));
		auto Renderable = EgyptianCat->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 0 | 1 << 1;
		Renderable->SetMesh(ModelMng.GetMesh(L"EgyptianCat:Cat_Statue_CatStatue"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Objects/EgyptianCat"));
		EgyptianCat->CreateComponent<CSceneProp>(0.4F)->GameStateComponent = GameState;
	}

	{
		auto FalloutCar = CreateObject<ESource::GGameObject>(L"FalloutCar", ESource::Transform(Vector3(80.F, -0.96F, 8.F), Quaternion::FromEulerAngles({ -74.F, 6.F, -143.F }), 1.F));
		auto PhysicsBody = FalloutCar->CreateComponent<ESource::CPhysicBody>();
		PhysicsBody->SetMesh(ModelMng.GetMesh(L"FalloutCar:default"));
		auto Renderable = FalloutCar->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 0 | 1 << 1;
		Renderable->SetMesh(ModelMng.GetMesh(L"FalloutCar:default"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Objects/FalloutCar"));
		FalloutCar->CreateComponent<CSceneProp>(0.2F)->GameStateComponent = GameState;
	}

	{
		auto Backpack = CreateObject<ESource::GGameObject>(L"Backpack", ESource::Transform(Vector3(0.F, 0.F, 0.F), Quaternion::FromEulerAngles({ -74.F, 6.F, -143.F }), 0.6F));
		auto PhysicsBody = Backpack->CreateComponent<ESource::CPhysicBody>();
		PhysicsBody->SetMesh(ModelMng.GetMesh(L"Backpack:Cylinder025"));
		auto Renderable = Backpack->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 0 | 1 << 1;
		Renderable->SetMesh(ModelMng.GetMesh(L"Backpack:Cylinder025"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Objects/Backpack"));
		Backpack->CreateComponent<CSceneProp>(1.F)->GameStateComponent = GameState;
	}

	{
		auto Prop = CreateObject<ESource::GGameObject>(L"Neko", ESource::Transform(Vector3(0.F, 0.F, 0.F), Quaternion::FromEulerAngles({ 0, 1.5F, -10.F }), 1.F));
		auto PhysicsBody = Prop->CreateComponent<ESource::CPhysicBody>();
		PhysicsBody->SetMesh(ModelMng.GetMesh(L"Neko:NekoCollision"));
		auto Renderable = Prop->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 0 | 1 << 1;
		Renderable->SetMesh(ModelMng.GetMesh(L"Neko:Neko"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Objects/NekoEye"));
		Renderable->SetMaterialAt(1, MaterialMng.GetMaterial(L"Objects/Neko"));
		Prop->AttachTo(PropCameraOffset);
		CProp * PropComponent = Prop->CreateComponent<CProp>();
		PropComponent->SetPlayerCamera(CameraComponent2);
		PropComponent->PhysicBody = PhysicsBody;
		PropComponent->GameStateComponent = GameState;
	}

	{
		auto Gun = CreateObject<ESource::GGameObject>(L"Gun", ESource::Transform(Vector3(-0.165F, -0.04F, 0.175F), Quaternion(), 0.01F));
		auto FlareGun = CreateObject<ESource::GGameObject>(L"FlareGun", ESource::Transform());
		auto Animator = FlareGun->CreateComponent<ESource::CAnimable>();
		Animator->Track = &ModelMng.GetModel(L"FlareGun")->GetAnimations()[0];
		auto FlareGunFrame = CreateObject<ESource::GGameObject>(L"FlareGun_Frame", ESource::Transform());
		ESource::CRenderable * Renderable = FlareGunFrame->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 0 | 1 << 1;
		Renderable->SetMesh(ModelMng.GetMesh(L"FlareGun:FlareGun_Frame"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Objects/FlareGun"));
		auto FlareGunBarrel = CreateObject<ESource::GGameObject>(L"FlareGun_Barrel", ESource::Transform());
		Renderable = FlareGunBarrel->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 0 | 1 << 1;
		Renderable->SetMesh(ModelMng.GetMesh(L"FlareGun:FlareGun_Barrel"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Objects/FlareGun"));
		auto FlareGunHammer = CreateObject<ESource::GGameObject>(L"FlareGun_Hammer", ESource::Transform());
		Renderable = FlareGunHammer->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 0 | 1 << 1;
		Renderable->SetMesh(ModelMng.GetMesh(L"FlareGun:FlareGun_Hammer"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Objects/FlareGun"));
		auto FlareGunTrigger = CreateObject<ESource::GGameObject>(L"FlareGun_Trigger", ESource::Transform());
		Renderable = FlareGunTrigger->CreateComponent<ESource::CRenderable>();
		Renderable->RenderingMask = 1 << 0 | 1 << 1;
		Renderable->SetMesh(ModelMng.GetMesh(L"FlareGun:FlareGun_Trigger"));
		Renderable->SetMaterialAt(0, MaterialMng.GetMaterial(L"Objects/FlareGun"));
		FlareGunFrame->AttachTo(FlareGun);
		FlareGunBarrel->AttachTo(FlareGun);
		FlareGunHammer->AttachTo(FlareGun);
		FlareGunTrigger->AttachTo(FlareGun);
		FlareGun->AttachTo(Gun);
		Gun->AttachTo(MainCamera);

		auto GunComponent = Gun->CreateComponent<CGun>();
		GunComponent->GameStateComponent = GameState;
		GunComponent->SetGunObjects(Gun, Animator, CameraComponent1);
	}
}

void GameSpaceLayer::OnRender() {
	ESource::Application::GetInstance()->GetRenderPipeline().Begin();
	Super::OnRender();
	ESource::Application::GetInstance()->GetRenderPipeline().End();
}

void GameSpaceLayer::OnPostRender() {
	ESource::Rendering::SetViewport(ESource::Application::GetInstance()->GetWindow().GetViewport());
	Super::OnPostRender();
}

ESource::GGameObject * ModelHierarchyToSpaceHierarchy(ESource::SpaceLayer * Space, ESource::RModel *& Model, ESource::ModelNode * Node, ESource::GGameObject * NewObject) {
	if (NewObject == NULL)
		NewObject = Space->CreateObject<ESource::GGameObject>(ESource::Text::NarrowToWide(Node->Name), Node->LocalTransform);

	if (Node->bHasMesh) {
		auto MeshRenderer = NewObject->CreateComponent<ESource::CRenderable>();
		MeshRenderer->SetMesh(Model->GetMeshes().at(Node->MeshKey));
		MeshRenderer->SetMaterialAt(0, ESource::MaterialManager::GetInstance().GetMaterial(L"Sponza/Bricks"));
	}
	for (auto & Child : Node->Children) {
		ESource::GGameObject * ChildGO = ModelHierarchyToSpaceHierarchy(Space, Model, Child, NULL);
		ChildGO->AttachTo(NewObject);
	}
	return NewObject;
}

void GameSpaceLayer::OnImGuiRender() {
	ImGui::Begin("Sandbox Space");

	TArray<ESource::IName> MaterialNameList = ESource::MaterialManager::GetInstance().GetResourceNames();
	TArray<NString> NarrowMaterialNameList(MaterialNameList.size());
	for (int i = 0; i < MaterialNameList.size(); ++i)
		NarrowMaterialNameList[i] = MaterialNameList[i].GetNarrowDisplayName();

	TArray<ESource::IName> MeshNameList = ESource::ModelManager::GetInstance().GetResourceMeshNames();
	TArray<NString> NarrowMeshNameList(MeshNameList.size());
	for (int i = 0; i < MeshNameList.size(); ++i)
		NarrowMeshNameList[i] = ESource::Text::WideToNarrow((MeshNameList)[i].GetDisplayName());

	static NChar Text[20];
	ImGui::InputText("##Renderer", Text, 20);
	ImGui::SameLine();
	if (ImGui::Button("Create GObject")) {
		if (strlen(Text) > 0) {
			CreateObject<ESource::GGameObject>(ESource::Text::NarrowToWide(NString(Text)), ESource::Transform(0.F, Quaternion(), 1.F));
		}
		Text[0] = '\0';
	}

	if (ImGui::BeginDragDropTarget()) {
		if (const ImGuiPayload* Payload = ImGui::AcceptDragDropPayload("ModelHierarchy")) {
			ES_ASSERT(Payload->DataSize == sizeof(ESource::RModel), "DragDropData is empty");
			ESource::RModel * PayloadModel = *(ESource::RModel**)Payload;
			LOG_CORE_WARN(L"Payload Model {}", PayloadModel->GetName().GetDisplayName());

			if (PayloadModel->GetHierarchyParentNode() != NULL) {
				ESource::GGameObject * NewGO = ModelHierarchyToSpaceHierarchy(
					this, PayloadModel, PayloadModel->GetHierarchyParentNode(), CreateObject<ESource::GGameObject>(PayloadModel->GetName().GetDisplayName())
				);

				if (PayloadModel->GetAnimations().size() > 0) {
					ESource::CAnimable * Animable = NewGO->CreateComponent<ESource::CAnimable>();
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

	TArray<ESource::GGameObject *> GameObjects;
	GetAllObjects<ESource::GGameObject>(GameObjects);
	for (auto & GameObject : GameObjects)
		if (GameObject->IsRoot()) {
			RenderGameObjectRecursive(GameObject, NarrowMaterialNameList, MaterialNameList, NarrowMeshNameList, MeshNameList, this);
		}

	ImGui::End();
}

GameSpaceLayer::GameSpaceLayer(const ESource::WString & Name, unsigned int Level) : SpaceLayer(Name, Level) {
}