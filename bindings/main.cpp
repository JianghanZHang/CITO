#include "mim-cio/python.hpp"
// #include "mim-cio/fwd.hpp"
#include <eigenpy/eigenpy.hpp>

BOOST_PYTHON_MODULE(mim_cio_pywrap) { 

    namespace bp = boost::python;

    bp::import("crocoddyl");

    mim_cio::python::exposeDAMmujoco();
    mim_cio::python::exposeIAMmujoco();
}
