
#include "../include/LogCore.h"
#include "../include/Object.h"

Object::Object() : IIdentifier() { 
	InternalName = L"object_" + std::to_wstring(IdentifierNum);
}

void Object::Delete() {
	SpaceIn->DestroyObject(this);
}

WString Object::GetName() {
	return InternalName;
}
