// This file contains forward declarations of the inherited classes
// Author: Jianghan Zhang
#ifndef cito_FWD_HPP_
#define cito_FWD_HPP_
#include <crocoddyl/core/utils/deprecate.hpp>
// This is necessary for Eigen to work with Boost.Python
#include <eigenpy/eigenpy.hpp>
namespace cito {
    template <typename Scalar>
    class IntegratedActionModelContactMjTpl;

    template <typename Scalar>
    struct IntegratedActionDataContactMjTpl;

    template <typename Scalar>
    class DifferentialActionModelContactMjTpl;

    template <typename Scalar>
    struct DifferentialActionDataContactMjTpl;

    typedef IntegratedActionModelContactMjTpl<double> IntegratedActionModelContactMj;
    typedef IntegratedActionDataContactMjTpl<double> IntegratedActionDataContactMj;
    typedef DifferentialActionModelContactMjTpl<double> DifferentialActionModelContactMj;
    typedef DifferentialActionDataContactMjTpl<double> DifferentialActionDataContactMj;

} // namespace cito

#endif // cito_FWD_HPP_