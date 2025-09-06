#pragma once
#include <string>

enum class LogLevel { INFO, WARNING, ERROR };

class Logger {
public:
    static void init();
    static void log(LogLevel level, const std::string& subsystem, const std::string& message);
    static void info(const std::string& subsystem, const std::string& message);
    static void warn(const std::string& subsystem, const std::string& message);
    static void error(const std::string& subsystem, const std::string& message);
};
