#pragma once

class IIdentifier {
public: 
	virtual size_t GetIdentifier();

	IIdentifier() {
		IdentifierNum = CurrentIdentifier++;
	}

private:
	// Identifier Generator
	static size_t CurrentIdentifier;

protected:
	size_t IdentifierNum;

};
