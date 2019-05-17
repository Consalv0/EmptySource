#pragma once

#include "../include/Object.h"
#include "../include/Transform.h"

class GameObject : public Object {
private:
	typedef Object Supper;
	friend class Space;

	GameObject();
	GameObject(const WString & Name);
	GameObject(const Transform & LocalTransform);
	GameObject(const WString & Name, const Transform & LocalTransform);

public:
	Transform SpaceMatrix;
};