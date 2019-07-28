#pragma once

#include "Text.h"
#include "Utility/Hasher.h"

class IIdentifier {
public: 
	IIdentifier();
	IIdentifier(const WString & Name);

	WString GetUniqueName() const;
	size_t GetUniqueID() const;

private:

	void ProcessIdentifierName(const WString & Name);

	WString InternalName;
	size_t NameHash;
};
