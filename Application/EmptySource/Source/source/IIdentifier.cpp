
#include <stddef.h>
#include "../include/IIdentifier.h"

size_t IIdentifier::CurrentIdentifier = 0;

size_t IIdentifier::GetIdentifier() const {
	return IdentifierNum;
}

WString IIdentifier::GetIdentifierName() const {
	return InternalName;
}

IIdentifier::IIdentifier() {
	IdentifierNum = CurrentIdentifier++;
}
