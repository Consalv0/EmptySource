
#include "..\include\LogCore.h"
#include "..\include\Space.h"
#include "..\include\Object.h"

TDictionary<size_t, Space*> Space::AllSpaces = TDictionary<size_t, Space*>();

Space::Space() : IIdentifier() {
	AllSpaces.insert(std::pair<const size_t, Space*>(GetIdentifier(), this));
}

Space::Space(Space & space) : IIdentifier() {
	AllSpaces.insert(std::pair<const size_t, Space*>(GetIdentifier(), this));
	ObjectsIn = TDictionary<size_t, Object*>();
}

size_t Space::GetIdentifier() {
	return IdentifierNum;
}

Space * Space::GetFirstSpace() {
	return AllSpaces.begin()->second;
}

Object * Space::MakeObject() {
	Object* NewObject = new Object();
	NewObject->SpaceIn = this;
	NewObject->InternalName = L"object_" + std::to_wstring(NewObject->IdentifierNum);
	Add(NewObject);
	return NewObject;
}

void Space::DestroyAllObjects() {
	for (TDictionary<size_t, Object*>::iterator Iterator = ObjectsIn.begin(); Iterator != ObjectsIn.end(); Iterator++) {
		Iterator->second->PrepareDelete();
	}
}

void Space::DestroyObject(Object * object) {
	ObjectsIn.erase(object->GetIdentifier());
}

void Space::Add(Object * object) {
	ObjectsIn.insert(std::pair<const size_t, Object*>(object->GetIdentifier(), object));
}
