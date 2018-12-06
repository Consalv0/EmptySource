
#include "../include/Object.h"
#include "../include/Space.h"

Object::Object() : IIdentifier() { }

Object * Object::MakeObject(Space * space) {
	Object* NewObject = new Object();
	space->Add(NewObject);
	NewObject->SpaceIn = space;
	return NewObject;
}

void Object::PrepareDelete() {
	SpaceIn->DestroyObject(this);
}
