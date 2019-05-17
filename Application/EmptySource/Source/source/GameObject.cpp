
#include "../include/GameObject.h"

GameObject::GameObject() : Object(L"GameObject") {
	Transformation = Transform();
}

GameObject::GameObject(const WString & Name) : Object(Name) {
	Transformation = Transform();
}

GameObject::GameObject(const Transform & LocalTransform) : Object(L"GameObject") {
	Transformation = LocalTransform;
}

GameObject::GameObject(const WString & Name, const Transform & LocalTransform) : Object(Name) {
	Transformation = LocalTransform;
}
