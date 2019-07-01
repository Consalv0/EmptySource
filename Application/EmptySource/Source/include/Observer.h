#pragma once

#include "../include/CoreTypes.h"
#include <functional>

struct Observer {
public:
	typedef Observer Supper;

	virtual void Call() const;

	int AddCallback(std::function<void()>);

	void RemoveCallback(const int& Identifier);

private:
	TDictionary<int, std::function<void()>> Callbacks;
};