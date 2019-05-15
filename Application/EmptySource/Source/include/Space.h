#pragma once

#include "CoreTypes.h"
#include "IIdentifier.h"

class Object;

class Space : public IIdentifier {
public:
	// Get first Space created
	static Space* GetFirstSpace();

	static Space* CreateSpace(const WString& Name);

	static void Destroy(Space * OtherSpace);

	// Destroys all objects in this Space
	void DestroyAllObjects();

	// Destroy specific object in this Space
	void DestroyObject(Object* object);

	WString GetName();

	// Creates an object in this space
	template<typename T>
	T * MakeObject() {
		T* NewObject = new T();
		Add(NewObject);
		return NewObject;
	}

private:
	Space();

	Space(const WString & Name);

	Space(Space& OtherSpace);
	
	WString Name;

	// Dictionary of all Spaces created
	static TDictionary<size_t, Space*> AllSpaces;

	// Dictionary that contains all the Objects in this Space
	TDictionary<size_t, Object*> ObjectsIn;

	// Add object in this space
	void Add(Object* object);
};