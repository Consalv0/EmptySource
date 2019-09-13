
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "..\..\Public\Core\GameObject.h"

namespace EmptySource {

	GGameObject::GGameObject() : OObject(L"GameObject"), Parent(NULL) {
		LocalTransform = Transform();
	}

	GGameObject::GGameObject(const WString & Name) : OObject(Name), Parent(NULL) {
		LocalTransform = Transform();
	}

	GGameObject::GGameObject(const Transform & InTransform) : OObject(L"GameObject"), Parent(NULL) {
		LocalTransform = InTransform;
	}

	GGameObject::GGameObject(const WString & Name, const Transform & InTransform) : OObject(Name), Parent(NULL) {
		LocalTransform = InTransform;
	}

	void GGameObject::AttachTo(GGameObject * InParent) {
		InParent->Children.push_back(this);
		Parent = InParent;
	}

	void GGameObject::DeatachFromParent() {
		for (TArray<GGameObject *>::const_iterator Iterator = Parent->Children.begin(); Iterator != Parent->Children.end(); ++Iterator)
			if ((*Iterator)->GetUniqueID() == GetUniqueID()) {
				Parent->Children.erase(Iterator);
				break;
			}
		Parent = NULL;
	}

	bool GGameObject::Contains(GGameObject * Other) const {
		for (auto & Child : Children)
			if (Child->GetUniqueID() == Other->GetUniqueID())
				return true;
		return false;
	}

	Transform GGameObject::GetWorldTransform() const {
		if (Parent == NULL) return LocalTransform;
		return LocalTransform * Parent->GetWorldTransform();
	}

	void GGameObject::AttachComponent(CComponent * Component) {
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