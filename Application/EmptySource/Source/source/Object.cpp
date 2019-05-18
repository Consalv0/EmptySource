
#include "../include/Core.h"
#include "../include/Object.h"

OObject::OObject() : IIdentifier(L"Object") { 
	Name = GetIdentifierName();
}

OObject::OObject(const WString & InName) : IIdentifier(InName) {
	Name = InName;
}

WString OObject::GetName() {
	return Name;
}
