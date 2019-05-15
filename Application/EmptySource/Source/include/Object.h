#pragma once

#include "../include/Space.h"
#include "../include/IIdentifier.h"

class Object : public IIdentifier {
private:
	friend class Space;

	Space* SpaceIn;

	Object();

	// Space initializer
	virtual void Initialize() {};

public:

	// Safe methos to delete this object removing it from the Space
	virtual void Delete();

	// Internal name of this object
	WString GetName();
};