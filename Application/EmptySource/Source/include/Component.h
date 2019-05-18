#pragma once

#include "../include/Object.h"
class GGameObject;

class CComponent : public OObject {
private:
	typedef OObject Supper;
	friend class GGameObject;
	friend class Space;

	CComponent(GGameObject & GameObject);

	GGameObject & Holder;

public:

	GGameObject & GetGameObject() const;

};