
#include "CoreMinimal.h"
#include "Core/Object.h"

namespace EmptySource {

	OObject::OObject() : Name(L"Object") {
		bAttached = false;
	}

	OObject::OObject(const IName & InName) : Name(InName) {
		bAttached = false;
	}

}