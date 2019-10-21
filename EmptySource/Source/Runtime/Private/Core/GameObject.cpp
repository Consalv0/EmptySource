
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "..\..\Public\Core\GameObject.h"

namespace ESource {

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
		if (InParent->ContainsRecursiveUp(this)) { 
			DeatachFromParent(); return; 
		}
		if (!IsRoot()) DeatachFromParent();
		InParent->Children.push_back(this);
		Parent = InParent;
	}

	void GGameObject::DeatachFromParent() {
		if (!IsRoot())
			for (TArray<GGameObject *>::const_iterator Iterator = Parent->Children.begin(); Iterator != Parent->Children.end(); ++Iterator)
				if ((*Iterator)->Name.GetInstanceID() == Name.GetInstanceID()) {
					Parent->Children.erase(Iterator);
					break;
				}
		Parent = NULL;
	}

	bool GGameObject::Contains(GGameObject * Other) const {
		for (auto & Child : Children)
			if (Child->Name.GetInstanceID() == Other->Name.GetInstanceID())
				return true;
		return false;
	}

	bool GGameObject::ContainsRecursiveDown(GGameObject * Other) const {
		for (auto & Child : Children)
			if (Child->Name.GetInstanceID() != Other->Name.GetInstanceID()) {
				if (Child->ContainsRecursiveDown(Other)) return true;
			}
			else {
				return true;
			}
		return false;
	}

	void GGameObject::DestroyComponent(CComponent * Component) {
		DetachComponent(Component);
	}

	bool GGameObject::ContainsRecursiveUp(GGameObject * Other) const {
		if (!IsRoot())
			if (Parent->Name.GetInstanceID() != Other->Name.GetInstanceID()) {
				if (Parent->ContainsRecursiveUp(Other)) return true;
			}
			else {
				return true;
			}
		return false;
	}

	Matrix4x4 GGameObject::GetWorldMatrix() const {
		if (Parent == NULL) return LocalTransform.GetLocalToWorldMatrix();
		return Parent->GetWorldMatrix() * LocalTransform.GetLocalToWorldMatrix();
	}

	Transform GGameObject::GetWorldTransform() const {
		return Transform(GetWorldMatrix());
	}

	void GGameObject::AttachComponent(CComponent * Component) {
		ES_CORE_ASSERT(ComponentsIn.find(Component->Name.GetInstanceID()) == ComponentsIn.end(), L"Trying to insert deleted or already existing component");
		ComponentsIn.emplace(Component->Name.GetInstanceID(), Component);
		Component->SpaceIn = SpaceIn;
		if (IsAttached()) Component->OnAttach();
		Component->OnAwake();
	}

	void GGameObject::DetachComponent(CComponent * Component) {
		Component->OnDetach();
		ComponentsIn[Component->Name.GetInstanceID()] = NULL;
		ComponentsOut.emplace(Component->Name.GetInstanceID(), Component);
	}

	void GGameObject::DeleteOutComponents() {
		for (auto & Iterator : ComponentsOut) {
			ComponentsIn.erase(Iterator.second->GetName().GetInstanceID());
			Iterator.second->OnDelete();
			delete Iterator.second;
		}
		
		ComponentsOut.clear();
	}

	void GGameObject::DeatachAllComponents() {
		for (TDictionary<size_t, CComponent*>::iterator Iterator = ComponentsIn.begin(); Iterator != ComponentsIn.end(); Iterator++) {
			if (Iterator->second)
				DetachComponent(Iterator->second);
		}
	}

	void GGameObject::OnRender() {
		for (auto & Component : ComponentsIn)
			Component.second->OnRender();
	}

	void GGameObject::OnUpdate(const Timestamp & Stamp) {
		DeleteOutComponents();
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
		DeatachAllComponents();
		DeleteOutComponents();
		TArray<GGameObject *> SChildren = Children;
		for (auto & Child : SChildren) {
			SpaceIn->DeleteObject(Child);
		}
		DeatachFromParent();
	}

}