#include "../include/IIdentifier.h"

size_t IIdentifier::CurrentIdentifier = 0;

size_t IIdentifier::GetIdentifier() {
	return IdentifierNum;
}
