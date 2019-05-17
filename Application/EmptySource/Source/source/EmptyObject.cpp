
#include "../include/Core.h"
#include "../include/EmptyObject.h"

EmptyObject::EmptyObject() : IIdentifier() { 
	InternalName = L"EmptyObject_" + std::to_wstring(IdentifierNum);
}

void EmptyObject::Delete() {
	SpaceIn->DestroyObject(this);
}

WString EmptyObject::GetName() {
	return InternalName;
}
