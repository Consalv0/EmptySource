#pragma once

#include "../include/Object.h"
#include "../include/Transform.h"
#include "../include/Component.h"

//* Basic class for any object that contains components and a spacial representation
class GGameObject : public OObject {
private:
	typedef OObject Supper;
	friend class Space;

	GGameObject();
	GGameObject(const WString & Name);
	GGameObject(const Transform & LocalTransform);
	GGameObject(const WString & Name, const Transform & LocalTransform);

	//* Dictionary that contains all the Components in this GameObject
	TDictionary<size_t, CComponent*> ComponentsIn;

	//* Add component
	void AddComponent(CComponent * Component);

	void DeleteComponent(CComponent * Component);
	void DeleteAllComponents();

	virtual void OnDelete();

public:
	Transform Transformation;

	TArray<GGameObject> Children;

	template<typename T, typename... Rest>
	CComponent * CreateComponent(Rest... Args);
};

template<typename T, typename ...Rest>
CComponent * GGameObject::CreateComponent(Rest ...Args) {
	T* NewObject = new T(*this, Args...);
	AddComponent(NewObject);
	return NewObject;
}