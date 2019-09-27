#pragma once

#include "CoreTypes.h"
#include <functional>

namespace ESource {

	struct Observer {
	public:
		typedef Observer Supper;

		virtual void Call() const;

		bool AddCallback(const NString& Identifier, std::function<void()>);

		void RemoveCallback(const NString& Identifier);

		void RemoveAllCallbacks();

		virtual ~Observer();

	private:
		TDictionary<NString, std::function<void()>> Callbacks;
	};

}