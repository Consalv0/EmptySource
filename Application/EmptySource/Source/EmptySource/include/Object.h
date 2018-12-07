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

	virtual void PrepareDelete();

	WString GetName();
};