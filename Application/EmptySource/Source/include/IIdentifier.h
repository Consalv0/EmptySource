#pragma once

#include "Text.h"
#include "Utility/Hasher.h"

class IIdentifier {
public: 
	IIdentifier();
	IIdentifier(const WString & Name);

	WString GetIdentifierName() const;
	size_t GetIdentifierHash() const;

private:
	// Identifier Generator
	static size_t CurrentIdentifier;
	WString InternalName;
	size_t NameHash;
};
