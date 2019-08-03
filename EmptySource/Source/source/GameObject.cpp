
#include "../include/GameObject.h"

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
	Component->Initialize();
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

void GGameObject::OnDelete() {
	DeleteAllComponents();
}
