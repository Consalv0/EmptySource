#pragma once

#include "CoreTypes.h"
#include "IIdentifier.h"

class Object;

class Space : public IIdentifier {
public:

	Space();
	Space(Space& space);

	// Get ID of this instance
	size_t GetIdentifier();

	// Get first Space created
	static Space* GetFirstSpace();

	// Destroys all objects in this Space
	void DestroyAllObjects();

	// Destroy specific object in this Space
	void DestroyObject(Object* object);

	void Add(Object* object);

private:

	// Dictionary of all Spaces loaded
	static TDictionary<size_t, Space*> AllSpaces;

	// Dictionary that contains all the Objects in this Space
	TDictionary<size_t, Object*> ObjectsIn;
};