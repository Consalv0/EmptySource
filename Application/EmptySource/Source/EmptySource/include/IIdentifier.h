#pragma once

class IIdentifier {
public: 
	virtual size_t GetIdentifier() = 0;

	IIdentifier() {
		IdentifierNum = CurrentIdentifier++;
	}

private:
	// Identifier Generator
	static size_t CurrentIdentifier;

protected:
	size_t IdentifierNum;

};
