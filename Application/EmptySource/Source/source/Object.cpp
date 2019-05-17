
#include "../include/Core.h"
#include "../include/Object.h"

Object::Object() : IIdentifier(L"Object") { 
	Name = GetIdentifierName();
}

Object::Object(const WString & InName) : IIdentifier(InName) {
	Name = InName;
}

void Object::Delete() {
	SpaceIn->DestroyObject(this);
}

WString Object::GetName() {
	return Name;
}
