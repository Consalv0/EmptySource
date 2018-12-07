
#include "../include/LogCore.h"
#include "../include/Object.h"

Object::Object() : IIdentifier() { }

void Object::PrepareDelete() {
	SpaceIn->DestroyObject(this);
}

WString Object::GetName() {
	return InternalName;
}
