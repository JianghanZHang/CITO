#ifndef __cito_python__
#define __cito_python__

#include <pinocchio/multibody/fwd.hpp>  // Must be included first!
#include <boost/python.hpp>

#include "cito/actions/differential-contact-mujoco.hpp"
#include "cito/actions/integrated-contact-mujoco.hpp"
#include "cito/actuations/floating-base-manipulation.hpp"
#include "cito/fwd.hpp"

namespace cito{
namespace python{
    void exposeDAMmujoco();
    void exposeIAMmujoco();
    void exposeActuationFloatingBaseManipulation();
} // namespace python
} // namespace cito

#endif