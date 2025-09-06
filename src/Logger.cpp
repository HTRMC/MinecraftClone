#include "Logger.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#undef ERROR
#endif

void Logger::init() {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    if (hOut != INVALID_HANDLE_VALUE && GetConsoleMode(hOut, &dwMode)) {
        SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#endif
}

static const char* colorForLevel(LogLevel level) {
    switch (level) {
        case LogLevel::INFO:    return "\033[32m"; // green
        case LogLevel::WARNING: return "\033[33m"; // yellow
        case LogLevel::ERROR:   return "\033[31m"; // red
        default:                return "\033[37m"; // white
    }
}

void Logger::log(LogLevel level, const std::string& subsystem, const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif

    std::cout << colorForLevel(level)
              << "[" << std::put_time(&tm, "%H:%M:%S") << "] "
              << "[" << subsystem << "] "
              << message
              << "\033[0m" << std::endl; // reset color
}

void Logger::info(const std::string& subsystem, const std::string& message) {
    log(LogLevel::INFO, subsystem, message);
}

void Logger::warn(const std::string& subsystem, const std::string& message) {
    log(LogLevel::WARNING, subsystem, message);
}

void Logger::error(const std::string& subsystem, const std::string& message) {
    log(LogLevel::ERROR, subsystem, message);
}
