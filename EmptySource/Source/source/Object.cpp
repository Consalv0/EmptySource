
#include "../include/Core.h"
#include "../include/Object.h"

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