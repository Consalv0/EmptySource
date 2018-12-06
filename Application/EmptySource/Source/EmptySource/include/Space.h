#pragma once

#include "CoreTypes.h"
#include "IIdentifier.h"

class Object;

class Space : public IIdentifier {
public:

	Space();

	size_t GetIdentifier();

private:

	// Dictionary that contains all the Objects in this Space
	TDictionary<size_t, Object> ObjectsIn;

	// Destroys all objects in this Space
	void DestroyAllObjects();

	// Destroy specific object in this Space
	void DestroyObject(Object* object);
};