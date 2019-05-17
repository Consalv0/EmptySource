
#include <stddef.h>
#include "../include/IIdentifier.h"

size_t IIdentifier::CurrentIdentifier = 0;

size_t IIdentifier::GetIdentifierHash() const {
	return NameHash;
}

WString IIdentifier::GetIdentifierName() const {
	return InternalName;
}

IIdentifier::IIdentifier() {
	InternalName = L"Identifier_" + std::to_wstring(++CurrentIdentifier);
	NameHash = GetHashName(InternalName);
}

IIdentifier::IIdentifier(const WString & Name) {
	InternalName = Name;
	if (!Text::ReplaceFromLast(InternalName, L"_", L"_" + std::to_wstring(++CurrentIdentifier))) {
		InternalName += L"_" + std::to_wstring(CurrentIdentifier);
	}
	NameHash = GetHashName(Name);
}
