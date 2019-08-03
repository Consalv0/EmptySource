#include "../include/ResourceHolder.h"

ResourceHolder::ResourceHolder(ResourceManager * Manager, const WString & Name) : 
	IIdentifier(Name), Manager(Manager), Name(Name), LoadState(LS_Unloaded) {
}
