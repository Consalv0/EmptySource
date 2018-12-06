#include "../include/Space.h"
#include "../include/Object.h"

Space::Space() : IIdentifier() {
}

size_t Space::GetIdentifier() {
	return IdentifierNum;
}

void Space::DestroyAllObjects() {
	for (TDictionary<size_t, Object>::iterator Iterator = ObjectsIn.begin(); Iterator != ObjectsIn.end(); Iterator++) {
		Iterator->second.DeletePrepare();
	}
}

void Space::DestroyObject(Object * object) {
}
