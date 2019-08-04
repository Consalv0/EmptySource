
#include "Resources/ResourceHolder.h"

namespace EmptySource {

	ResourceHolder::ResourceHolder(ResourceManager * Manager, const WString & Name) :
		IIdentifier(Name), Manager(Manager), Name(Name), LoadState(LS_Unloaded) {
	}

}