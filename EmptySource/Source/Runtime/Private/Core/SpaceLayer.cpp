
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "Core/SpaceLayer.h"

#include "Utility/TextFormatting.h"

namespace EmptySource {
	TDictionary<size_t, SpaceLayer*> SpaceLayer::AllSpaces = TDictionary<size_t, SpaceLayer*>();

	SpaceLayer::SpaceLayer(const WString& InName, unsigned int Level) : Layer(InName, Level) {
		bAttached = false;
		ObjectsIn = TDictionary<size_t, OObject*>();
		AllSpaces.insert(std::pair<const size_t, SpaceLayer*>(GetUniqueID(), this));
	}

	// SpaceLayer::SpaceLayer(SpaceLayer & OtherSpace) : Layer(OtherSpace.GetUniqueName(), OtherSpace.GetLayerPriority()) {
	// 	WString Number, Residue;
	// 	if (Text::GetLastNotOf(OtherSpace.Name, Residue, Number, L"0123456789"))
	// 		Name = Residue + std::to_wstring(std::stoi(Number) + 1);
	// 	else
	// 		Name = OtherSpace.Name + L"_1";
	// 
	// 	ObjectsIn = TDictionary<size_t, OObject*>();
	// 	AllSpaces.insert(std::pair<const size_t, SpaceLayer*>(GetUniqueID(), this));
	// }

	void SpaceLayer::OnAttach() {
		bAttached = true;
		for (TDictionary<size_t, OObject*>::iterator Iterator = ObjectsIn.begin(); Iterator != ObjectsIn.end(); Iterator++)
			Iterator->second->OnAttach();
	}

	void SpaceLayer::OnAwake() {
		TArray<OObject *> Objects;
		GetAllObjects<OObject>(Objects);
		for (TDictionary<size_t, OObject*>::iterator Iterator = ObjectsIn.begin(); Iterator != ObjectsIn.end(); Iterator++)
			Iterator->second->OnAwake();
	}

	void SpaceLayer::OnDetach() {
	}

	void SpaceLayer::OnRender() {
		TArray<GGameObject *> GameObjects;
		GetAllObjects<GGameObject>(GameObjects);
		for (auto & GameObject : GameObjects)
			GameObject->OnRender();
	}

	void SpaceLayer::OnUpdate(Timestamp Stamp) {
		TArray<GGameObject *> GameObjects;
		GetAllObjects<GGameObject>(GameObjects);
		for (auto & GameObject : GameObjects)
			GameObject->OnUpdate(Stamp);
	}

	void SpaceLayer::OnImGuiRender() {
		TArray<GGameObject *> GameObjects;
		GetAllObjects<GGameObject>(GameObjects);
		for (auto & GameObject : GameObjects)
			GameObject->OnImGuiRender();
	}

	void SpaceLayer::OnWindowEvent(WindowEvent & WinEvent) {
		TArray<GGameObject *> GameObjects;
		GetAllObjects<GGameObject>(GameObjects);
		for (auto & GameObject : GameObjects)
			GameObject->OnWindowEvent(WinEvent);
	}

	void SpaceLayer::OnInputEvent(InputEvent & InEvent) {
		TArray<GGameObject *> GameObjects;
		GetAllObjects<GGameObject>(GameObjects);
		for (auto & GameObject : GameObjects)
			GameObject->OnInputEvent(InEvent);
	}

	SpaceLayer::~SpaceLayer() {
		DeleteAllObjects();
		AllSpaces.erase(GetUniqueID());
		LOG_CORE_DEBUG(L"Space {} deleted", GetUniqueName().c_str());
	}

	SpaceLayer * SpaceLayer::GetSpace(const size_t & Identifier) {
		auto Find = AllSpaces.find(Identifier);
		if (AllSpaces.find(Identifier) == AllSpaces.end())
			return NULL;

		return Find->second;
	}

	WString SpaceLayer::GetFriendlyName() const {
		return Name;
	}

	OObject * SpaceLayer::GetObjectByID(size_t Identifier) {
		auto Find = ObjectsIn.find(Identifier);
		if (Find != ObjectsIn.end())
			return Find->second;
		
		return NULL;
	}

	void SpaceLayer::DeleteAllObjects() {
		for (TDictionary<size_t, OObject*>::iterator Iterator = ObjectsIn.begin(); Iterator != ObjectsIn.end(); Iterator++) {
			DeleteObject(Iterator->second);
		}
	}

	void SpaceLayer::DeleteObject(OObject * Object) {
		Object->OnDelete();
		ObjectsIn.erase(Object->GetUniqueID());
		delete Object;
	}

	void SpaceLayer::AddObject(OObject * Object) {
		Object->SpaceIn = this;
		ObjectsIn.insert(std::pair<const size_t, OObject*>(Object->GetUniqueID(), Object));
		if (bAttached) Object->OnAttach();
		Object->OnAwake();
	}

}