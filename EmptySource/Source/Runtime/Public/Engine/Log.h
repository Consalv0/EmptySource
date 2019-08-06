#pragma once

#include "Engine/Core.h"

#define SPDLOG_WCHAR_TO_UTF8_SUPPORT
#define SPDLOG_NO_DATETIME
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace EmptySource {

	class Log {
	public:
		static void Initialize();

		inline static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return CoreLogger; }
		inline static std::shared_ptr<spdlog::logger>& GetClientLogger() { return ClientLogger; }
	private:
		static std::shared_ptr<spdlog::logger> CoreLogger;
		static std::shared_ptr<spdlog::logger> ClientLogger;
	};

}

// Core log macros
#define LOG_CORE_DEBUG(...)    ::EmptySource::Log::GetCoreLogger()->debug(__VA_ARGS__)
#define LOG_CORE_INFO(...)     ::EmptySource::Log::GetCoreLogger()->info(__VA_ARGS__)
#define LOG_CORE_WARN(...)     ::EmptySource::Log::GetCoreLogger()->warn(__VA_ARGS__)
#define LOG_CORE_ERROR(...)    ::EmptySource::Log::GetCoreLogger()->error(__VA_ARGS__)
#define LOG_CORE_CRITICAL(...) ::EmptySource::Log::GetCoreLogger()->critical(__VA_ARGS__)

// Client log macros
#define LOG_DEBUG(...)         ::EmptySource::Log::GetClientLogger()->debug(__VA_ARGS__)
#define LOG_INFO(...)          ::EmptySource::Log::GetClientLogger()->info(__VA_ARGS__)
#define LOG_WARN(...)          ::EmptySource::Log::GetClientLogger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)         ::EmptySource::Log::GetClientLogger()->error(__VA_ARGS__)
#define LOG_CRITICAL(...)      ::EmptySource::Log::GetClientLogger()->critical(__VA_ARGS__)