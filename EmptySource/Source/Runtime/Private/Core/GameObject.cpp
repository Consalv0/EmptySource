
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "..\..\Public\Core\GameObject.h"

namespace EmptySource {

	GGameObject::GGameObject() : OObject(L"GameObject") {
		Transformation = Transform();
	}

	GGameObject::GGameObject(const WString & Name) : OObject(Name) {
		Transformation = Transform();
	}

	GGameObject::GGameObject(const Transform & LocalTransform) : OObject(L"GameObject") {
		Transformation = LocalTransform;
	}

	GGameObject::GGameObject(const WString & Name, const Transform & LocalTransform) : OObject(Name) {
		Transformation = LocalTransform;
	}

	void GGameObject::AddComponent(CComponent * Component) {
		ComponentsIn.insert(std::pair<const size_t, CComponent*>(Component->GetUniqueID(), Component));
		Component->SpaceIn = SpaceIn;
		if (IsAttached()) Component->OnAttach();
		Component->OnAwake();
	}

	void GGameObject::DeleteComponent(CComponent * Component) {
		Component->OnDelete();
		ComponentsIn.erase(Component->GetUniqueID());
		delete Component;
	}

	void GGameObject::DeleteAllComponents() {
		for (TDictionary<size_t, CComponent*>::iterator Iterator = ComponentsIn.begin(); Iterator != ComponentsIn.end(); Iterator++)
			DeleteComponent(Iterator->second);
	}

	void GGameObject::OnRender() {
		for (auto & Component : ComponentsIn)
			Component.second->OnRender();
	}

	void GGameObject::OnUpdate(const Timestamp & Stamp) {
		for (auto & Component : ComponentsIn)
			Component.second->OnUpdate(Stamp);
	}

	void GGameObject::OnImGuiRender() {
		for (auto & Component : ComponentsIn)
			Component.second->OnImGuiRender();
	}

	void GGameObject::OnWindowEvent(WindowEvent & WinEvent) {
		for (auto & Component : ComponentsIn)
			Component.second->OnWindowEvent(WinEvent);
	}

	void GGameObject::OnInputEvent(InputEvent & InEvent) {
		for (auto & Component : ComponentsIn)
			Component.second->OnInputEvent(InEvent);
	}

	void GGameObject::OnDelete() {
		DeleteAllComponents();
	}

}