
#include "Engine/Core.h"
#include "Engine/Object.h"

namespace EmptySource {

	OObject::OObject() : IIdentifier(L"Object") {
		Name = GetUniqueName();
	}

	OObject::OObject(const WString & InName) : IIdentifier(InName) {
		Name = InName;
	}

	WString OObject::GetName() {
		return Name;
	}

}