#pragma once

#include "../include/CoreTypes.h"
#include "../include/Text.h"
#include <functional>

struct Observer {
public:
	typedef Observer Supper;

	virtual void Call() const;

	bool AddCallback(const String& Identifier, std::function<void()>);

	void RemoveCallback(const String& Identifier);

	void RemoveAllCallbacks();

	virtual ~Observer();

private:
	TDictionary<String, std::function<void()>> Callbacks;
};