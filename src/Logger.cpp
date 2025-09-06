#include "Logger.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#undef ERROR
#endif

std::unordered_map<std::string, int> Logger::messageCount;
std::mutex Logger::messageCountMutex;

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
        case LogLevel::DEBUG:   return "\033[95m"; // bright magenta
        default:                return "\033[37m"; // white
    }
}

void Logger::log(LogLevel level, const std::string& subsystem, const std::string& message) {
#ifdef NDEBUG
    if (level == LogLevel::DEBUG) {
        return;
    }
#endif

    if (level == LogLevel::DEBUG && !shouldLogMessage(message)) {
        return;
    }
    
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

void Logger::warning(const std::string& subsystem, const std::string& message) {
    log(LogLevel::WARNING, subsystem, message);
}

void Logger::error(const std::string& subsystem, const std::string& message) {
    log(LogLevel::ERROR, subsystem, message);
}

void Logger::debug(const std::string& subsystem, const std::string& message) {
    log(LogLevel::DEBUG, subsystem, message);
}

bool Logger::shouldLogMessage(const std::string& message) {
    std::lock_guard<std::mutex> lock(messageCountMutex);
    
    auto& count = messageCount[message];
    count++;
    
    if (count <= MAX_REPEAT_MESSAGES) {
        return true;
    } else if (count == MAX_REPEAT_MESSAGES + 1) {
        std::cout << "\033[95m[DEBUG] Message suppressed: \"" << message 
                  << "\" (shown " << MAX_REPEAT_MESSAGES << " times)\033[0m" << std::endl;
    }
    
    return false;
}
