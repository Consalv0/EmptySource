
#include "../include/GameObject.h"
#include "../include/Component.h"

CComponent::CComponent(GGameObject & GameObject) : OObject(L"Component"), Holder(GameObject) {
}

GGameObject & CComponent::GetGameObject() const {
	return Holder;
}
