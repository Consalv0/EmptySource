#pragma once

#include "../include/Space.h"
#include "../include/IIdentifier.h"

class EmptyObject : public IIdentifier {
private:
	friend class Space;

	Space* SpaceIn;

	EmptyObject();

	// Space initializer
	virtual void Initialize() {};

public:

	// Safe methos to delete this object removing it from the Space
	virtual void Delete();

	// Internal name of this object
	WString GetName();
};