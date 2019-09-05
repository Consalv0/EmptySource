
#include "CoreMinimal.h"
#include "Core/Object.h"

namespace EmptySource {

	OObject::OObject() : IIdentifier(L"Object") {
		Name = GetUniqueName();
		bAttached = false;
	}

	OObject::OObject(const WString & InName) : IIdentifier(InName) {
		Name = InName;
		bAttached = false;
	}

	WString OObject::GetName() {
		return Name;
	}

}