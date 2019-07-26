#pragma once

#include "../include/Space.h"
#include "../include/IIdentifier.h"

class OObject : public IIdentifier {
protected:
	friend class Space;

	Space* SpaceIn;

	WString Name;

	OObject();
	OObject(const WString& Name);

	// Space initializer
	virtual bool Initialize() { return true; };

public:

	// Safe methos to delete this object removing it from the Space
	virtual void OnDelete() {};

	// Name of this object
	WString GetName();
};