#pragma once

#ifdef EMPTYSOURCE_CORE
#pragma message ( "You have placed Core/Log.h before Core.h. This may cause errors in ftm headers" )
#else 
#define EMPTYSOURCE_CORE_LOG
#endif // EMPTYSOURCE_CORE

#include "Core.h"
#define SPDLOG_WCHAR_TO_UTF8_SUPPORT
#define SPDLOG_NO_DATETIME
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace ESource {

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
#define LOG_CORE_DEBUG(...)    ::ESource::Log::GetCoreLogger()->debug(__VA_ARGS__)
#define LOG_CORE_INFO(...)     ::ESource::Log::GetCoreLogger()->info(__VA_ARGS__)
#define LOG_CORE_WARN(...)     ::ESource::Log::GetCoreLogger()->warn(__VA_ARGS__)
#define LOG_CORE_ERROR(...)    ::ESource::Log::GetCoreLogger()->error(__VA_ARGS__)
#define LOG_CORE_CRITICAL(...) ::ESource::Log::GetCoreLogger()->critical(__VA_ARGS__)

// Client log macros
#define LOG_DEBUG(...)         ::ESource::Log::GetClientLogger()->debug(__VA_ARGS__)
#define LOG_INFO(...)          ::ESource::Log::GetClientLogger()->info(__VA_ARGS__)
#define LOG_WARN(...)          ::ESource::Log::GetClientLogger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)         ::ESource::Log::GetClientLogger()->error(__VA_ARGS__)
#define LOG_CRITICAL(...)      ::ESource::Log::GetClientLogger()->critical(__VA_ARGS__)