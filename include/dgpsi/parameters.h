#ifndef PARAMETERS_H
#define PARAMETERS_H
#include <dgpsi/base.h>

namespace dgpsi::parameters {

    /* Constraints */
    template<typename T>
    T transform_fxn(const T& value, std::string& method, bool inverse = false) { T a; return a; };

    template<>
    TVector transform_fxn(const TVector& value, std::string& method, bool inverse) {
        if (method == "logexp") {
            if (inverse) { return log(value.array()); }
            else { return exp(value.array()); }
        }
        else { return value; }
    }

    template<>
    double transform_fxn(const double& value, std::string& method, bool inverse) {
        if (method == "logexp") {
            if (inverse) { return log(value); }
            else { return exp(value); }
        }
        else { return value; }
    }


    template<typename T>
    class BaseParameter {
    protected:
        T value_;
        std::string name_;
        std::pair<T, T> bounds_;
        std::size_t size_ = 1;
        bool fixed_ = false;
        std::string transform_ = "logexp";
    public:
        const bool* is_fixed = &fixed_;
    public:
        BaseParameter() = default;
        BaseParameter(const BaseParameter&) = default;
        BaseParameter(BaseParameter&&) = default;
        BaseParameter(std::string name, T value, std::string transform, std::pair<T, T> bounds) : name_(name), transform_(transform), value_(value), bounds_(bounds) {}

        virtual void fix() { fixed_ = true; is_fixed = &fixed_; }
        virtual void unfix() { fixed_ = false; is_fixed = &fixed_; }
        virtual void set_constraint(std::string constraint) {
            if (constraint == "none" || constraint == "logexp") {
                transform_ = constraint;
            }
            else { throw std::runtime_error("Unrecognized Constraint"); }
        }

        // ========== C++ Interface ========== //
        const std::size_t& size() { return size_; }
        virtual void transform_value(bool inverse = false) {
            value_ = transform_fxn(value_, transform_, inverse);
        }
        virtual void transform_value(const T& new_value, bool inverse = false) {
            value_ = transform_fxn(new_value, transform_, inverse);
        }
        virtual void transform_bounds(bool inverse = false) {
            T tmp1, tmp2;
            tmp1 = transform_fxn(bounds_.first, transform_, inverse);
            tmp2 = transform_fxn(bounds_.second, transform_, inverse);
            bounds_ = std::make_pair(tmp1, tmp2);
        }
        virtual void transform_bounds(const std::pair<T, T>& new_bounds, bool inverse = false) {
            T tmp1, tmp2;
            tmp1 = transform_fxn(new_bounds.first, transform_, inverse);
            tmp2 = transform_fxn(new_bounds.second, transform_, inverse);
            bounds_ = std::make_pair(tmp1, tmp2);
        }

        BaseParameter& operator=(const T& oValue) {
            if (fixed_) { throw std::runtime_error("Error fixed value.."); }
            value_ = oValue;
            return *this;
        }
        BaseParameter& operator=(const BaseParameter& oParam) {
            size_ = oParam.size_;
            name_ = oParam.name_;
            value_ = oParam.value_;
            fixed_ = oParam.fixed_;
            transform_ = oParam.transform_;
            bounds_ = oParam.bounds_;
            return *this;
        }
        template <typename U>
        friend std::ostream& operator<<(std::ostream& os, const BaseParameter<U>& param);
        // ================================== //

        // Setters
        virtual void set_value(const T& new_value) {
            if (fixed_) { throw std::runtime_error("Error fixed value.."); }
            value_ = new_value;
        };
        virtual void set_name(const std::string& new_name) { name_ = new_name; };
        virtual void set_transform(const std::string& new_transform) { transform_ = new_transform; };
        virtual void set_bounds(const std::pair<T, T>& new_bounds) = 0;
        virtual T value() const { return value_; }
        virtual const std::string name() const { return name_; }
        // Getters
        virtual const T get_value() const { return value_; };
        virtual const std::string get_name() const { return name_; }
        virtual const std::string get_transform() const { return transform_; }
        virtual const std::pair<T, T> get_bounds() const { return bounds_; }
        virtual const bool fixed() const { return fixed_; }

    };

    template<typename T>
    std::ostream& operator<<(std::ostream& stream, const BaseParameter<T>& param) {
        return stream << param.value;
    }


    template <typename T>
    class Parameter;

    template<>
    class Parameter<double> : public BaseParameter<double>
    {

    public:

        Parameter() : BaseParameter() {}
        Parameter(std::string name, double value) {
            name_ = name;
            transform_ = "logexp";
            value_ = value;
            bounds_ = std::make_pair(-std::numeric_limits<double>::infinity() , std::numeric_limits<double>::infinity());
        }
        Parameter(std::string name, double value, std::pair<double, double> bounds) {
            name_ = name;
            transform_ = "logexp";
            value_ = value;
            bounds_ = bounds;
        }
        Parameter(std::string name, double value, std::string transform) {
            name_ = name;
            transform_ = transform;
            value_ = value;
            bounds_ = std::make_pair(-std::numeric_limits<double>::infinity() , std::numeric_limits<double>::infinity());
        }
        Parameter(std::string name, double value, std::string transform, std::pair<double, double> bounds) : BaseParameter(name, value, transform, bounds) {}

        void set_bounds(const std::pair<double, double>& new_bounds) override {
            if (bounds_.first > bounds_.second) { throw std::runtime_error("Lower Bound > Upper Bound"); }
            bounds_ = new_bounds;
        }

        // C++ Interface
        Parameter& operator=(const double& oValue)
        {
            if (fixed_) { throw std::runtime_error("Error fixed value.."); }
            value_ = oValue;
            return *this;
        }

    };

    template<>
    class Parameter<TVector> : public BaseParameter<TVector>
    {
    protected:

    public:
        Parameter() : BaseParameter() {}
        Parameter(std::string name, TVector value) {
            name_ = name;
            transform_ = "logexp";
            value_ = value;
            size_ = value.size();
            TVector lower_bound(size_);
            TVector upper_bound(size_);
            for (int i = 0; i < size_; ++i) { lower_bound[i] = -std::numeric_limits<double>::infinity(); }
            for (int i = 0; i < size_; ++i) { upper_bound[i] = std::numeric_limits<double>::infinity(); }
            bounds_ = std::make_pair(lower_bound, upper_bound);
        }
        Parameter(std::string name, TVector value, std::string transform) {
            name_ = name;
            transform_ = transform;
            value_ = value;
            size_ = value.size();
            TVector lower_bound(size_);
            TVector upper_bound(size_);
            for (int i = 0; i < size_; ++i) { lower_bound[i] = -std::numeric_limits<double>::infinity();; }
            for (int i = 0; i < size_; ++i) { upper_bound[i] = std::numeric_limits<double>::infinity();; }
            bounds_ = std::make_pair(lower_bound, upper_bound);
        }
        Parameter(std::string name, TVector value, std::pair<TVector, TVector> bounds) {
            name_ = name;
            transform_ = "logexp";
            value_ = value;
            size_ = value.size();
            bounds_ = bounds;
        }
        Parameter(std::string name, TVector value, std::string transform, std::pair<TVector, TVector> bounds) : BaseParameter(name, value, transform, bounds) { size_ = value.size(); }

        void set_bounds(const std::pair<TVector, TVector>& new_bounds) override {
            if ((bounds_.first.array() > bounds_.second.array()).any())
            {
                throw std::runtime_error("Lower Bound > Upper Bound");
            }
            bounds_ = new_bounds;
        }

        // C++ Interface
        Parameter& operator=(const TVector& oValue)
        {
            if (fixed_) { throw std::runtime_error("Error fixed value.."); }
            value_ = oValue;
            size_ = value_.rows();
            return *this;
        }
        double& operator[](const Eigen::Index& idx) { return value_[idx]; }

    };
}
#endif