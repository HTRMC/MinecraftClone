#pragma once
#include <functional>
#include <string>

template<typename T>
class SimpleOption {
public:
    using ChangeCallback = std::function<void(const T&)>;

    SimpleOption(const std::string& name, const T& defaultValue, ChangeCallback onChange = nullptr);

    const T& getValue() const;
    void setValue(const T& newValue);

    const std::string& getName() const;

private:
    std::string name_;
    T value_;
    T defaultValue_;
    ChangeCallback onChange_;
};
