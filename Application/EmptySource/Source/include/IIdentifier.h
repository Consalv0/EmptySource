#pragma once

#include "Text.h"

class IIdentifier {
public: 

	IIdentifier();

private:
	// Identifier Generator
	static size_t CurrentIdentifier;

protected:
	WString InternalName;
	virtual WString GetIdentifierName() const;

	size_t IdentifierNum;
	size_t GetIdentifier() const;
};
