#include "SimpleOption.hpp"
#include <iostream>

template<typename T>
SimpleOption<T>::SimpleOption(const std::string& name, const T& defaultValue, ChangeCallback onChange)
    : name_(name), value_(defaultValue), defaultValue_(defaultValue), onChange_(onChange) {}

template<typename T>
const T& SimpleOption<T>::getValue() const {
    return value_;
}

template<typename T>
void SimpleOption<T>::setValue(const T& newValue) {
    if (newValue != value_) {
        value_ = newValue;
        if (onChange_) onChange_(value_);
    }
}

template<typename T>
const std::string& SimpleOption<T>::getName() const {
    return name_;
}

template class SimpleOption<int>;
