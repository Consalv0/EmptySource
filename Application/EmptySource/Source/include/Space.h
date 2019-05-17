#pragma once

#include "CoreTypes.h"
#include "IIdentifier.h"

class Object;

class Space : public IIdentifier {
public:
	// Get active main Space
	static Space* GetMainSpace();

	// Get Space by IIdentifier
	static Space* GetSpace(const size_t& Identifier);

	static Space* CreateSpace(const WString& Name);

	static void Destroy(Space * OtherSpace);

	// Destroys all objects in this Space
	void DestroyAllObjects();

	// Destroy specific object in this Space
	void DestroyObject(Object* object);

	WString GetName();

	// Creates an object in this space
	template<typename T>
	T * CreateObject() {
		T* NewObject = new T();
		Add(NewObject);
		return NewObject;
	}

	// Creates an object with name in this space
	template<typename T, typename... Rest>
	T * CreateObject(Rest... Args) {
		T* NewObject = new T(Args...);
		Add(NewObject);
		return NewObject;
	}

private:
	Space();

	Space(const WString & Name);

	Space(Space& OtherSpace);
	
	WString Name;

	// Dictionary that contains all the Objects in this Space
	TDictionary<size_t, Object*> ObjectsIn;

	// Add object in this space
	void Add(Object* Object);

protected:

	// Dictionary of all Spaces created
	static TDictionary<size_t, Space*> AllSpaces;
};