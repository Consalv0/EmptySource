
#include "CoreMinimal.h"
#include "Resources/ResourceHolder.h"

namespace ESource {

	ResourceHolder::ResourceHolder(const IName & Name, const WString& Origin) :
		Name(Name), LoadState(LS_Unloaded), Origin(Origin) {
	}

}