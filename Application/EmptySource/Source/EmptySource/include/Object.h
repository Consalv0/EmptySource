#pragma once

#include "../include/Space.h"
#include "../include/IIdentifier.h"

class Object : public IIdentifier {
private:
	friend class Space;

	WString InternalName;

	Space* SpaceIn;

	Object();

public:

	// Safe methos to delete this object removing it from the Space
	virtual void Delete();

	// Internal name of this object
	WString GetName();
};