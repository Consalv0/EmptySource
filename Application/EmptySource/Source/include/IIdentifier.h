#pragma once

class IIdentifier {
public: 
	virtual size_t GetIdentifier();

	IIdentifier();

private:
	// Identifier Generator
	static size_t CurrentIdentifier;

protected:
	size_t IdentifierNum;

};
