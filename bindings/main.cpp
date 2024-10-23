#include "cito/python.hpp"
#include "cito/fwd.hpp"
#include <eigenpy/eigenpy.hpp>

BOOST_PYTHON_MODULE(cito_pywrap) { 

    namespace bp = boost::python;

    bp::import("crocoddyl");

    cito::python::exposeDAMmujoco();
    cito::python::exposeIAMmujoco();
}
