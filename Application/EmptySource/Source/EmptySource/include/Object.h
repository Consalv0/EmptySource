#pragma once

#include "../include/IIdentifier.h"

class Space;

class Object : public IIdentifier {
private:
	Space* SpaceIn;

	Object();

public:

	static Object* MakeObject(Space* space);

	virtual void PrepareDelete();
};