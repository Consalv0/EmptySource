
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "Core/SpaceLayer.h"

#include "Utility/TextFormatting.h"

namespace ESource {
	TDictionary<size_t, SpaceLayer*> SpaceLayer::AllSpaces = TDictionary<size_t, SpaceLayer*>();

	SpaceLayer::SpaceLayer(const IName& InName, unsigned int Level) : Layer(InName, Level) {
		bAttached = false;
		ObjectsIn = TDictionary<size_t, OObject*>();
		AllSpaces.insert(std::pair<const size_t, SpaceLayer*>(Name.GetInstanceID(), this));
	}

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
		DeleteOutObjects();
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
		DeleteOutObjects();
		AllSpaces.erase(Name.GetInstanceID());
		LOG_CORE_DEBUG(L"Space {} deleted", Name.GetInstanceName().c_str());
	}

	SpaceLayer * SpaceLayer::GetSpace(const size_t & Identifier) {
		auto Find = AllSpaces.find(Identifier);
		if (AllSpaces.find(Identifier) == AllSpaces.end())
			return NULL;

		return Find->second;
	}

	const IName & SpaceLayer::GetName() const {
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
		ObjectsIn[Object->Name.GetInstanceID()] = NULL;
		ObjectsOut.emplace(Object->Name.GetInstanceID(), Object);
	}

	void SpaceLayer::AddObject(OObject * Object) {
		Object->SpaceIn = this;
		ObjectsIn.insert(std::pair<const size_t, OObject*>(Object->Name.GetInstanceID(), Object));
		if (bAttached) Object->OnAttach();
		Object->OnAwake();
	}

	void SpaceLayer::DeleteOutObjects() {
		for (auto & Iterator : ObjectsOut) {
			ObjectsIn.erase(Iterator.second->GetName().GetInstanceID());
			Iterator.second->OnDelete();
			delete Iterator.second;
		}

		ObjectsOut.clear();
	}

}