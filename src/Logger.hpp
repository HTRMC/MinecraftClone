#pragma once
#include <string>
#include <unordered_map>
#include <mutex>

enum class LogLevel { INFO, WARNING, ERROR, DEBUG };

class Logger {
public:
    static void init();
    static void log(LogLevel level, const std::string& subsystem, const std::string& message);
    static void info(const std::string& subsystem, const std::string& message);
    static void warning(const std::string& subsystem, const std::string& message);
    static void error(const std::string& subsystem, const std::string& message);
    static void debug(const std::string& subsystem, const std::string& message);
    
private:
    static bool shouldLogMessage(const std::string& message);
    static std::unordered_map<std::string, int> messageCount;
    static std::mutex messageCountMutex;
    static constexpr int MAX_REPEAT_MESSAGES = 10;
};
