#include "Logger.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

void Logger::info(const std::string& subsystem, const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::cout << "[" 
              << std::put_time(&tm, "%H:%M:%S") 
              << "] [" << subsystem << "/INFO]: " 
              << message << std::endl;
}
