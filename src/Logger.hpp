#pragma once
#include <string>

class Logger {
public:
    static void info(const std::string& subsystem, const std::string& message);
};
