
#include "../include/GameObject.h"

GameObject::GameObject() : Object(L"GameObject") {
	SpaceMatrix = Transform();
}

GameObject::GameObject(const WString & Name) : Object(Name) {
	SpaceMatrix = Transform();
}

GameObject::GameObject(const Transform & LocalTransform) : Object(L"GameObject") {
	SpaceMatrix = LocalTransform;
}

GameObject::GameObject(const WString & Name, const Transform & LocalTransform) : Object(Name) {
	SpaceMatrix = LocalTransform;
}
