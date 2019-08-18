
#include "CoreMinimal.h"
#include "Resources/ResourceHolder.h"

namespace EmptySource {

	Resource::Resource(const WString & Name) :
		Name(Name), UID(WStringToHash(Name)) {
	}

}