
#include "..\include\Core.h"
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
	Add(NewObject);
	return NewObject;
}

void Space::DestroyAllObjects() {
	for (TDictionary<size_t, Object*>::iterator Iterator = ObjectsIn.begin(); Iterator != ObjectsIn.end(); Iterator++) {
		Iterator->second->Delete();
	}
}

void Space::DestroyObject(Object * object) {
	ObjectsIn.erase(object->GetIdentifier());
	delete object;
}

void Space::Add(Object * object) {
	ObjectsIn.insert(std::pair<const size_t, Object*>(object->GetIdentifier(), object));
	object->SpaceIn = this;
}
