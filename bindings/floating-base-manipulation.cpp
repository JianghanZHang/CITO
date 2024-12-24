#include "cito/python.hpp"
#include "cito/fwd.hpp"
#include "utils/copyable.hpp"
#include "cito/actuations/floating-base-manipulation.hpp"
#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/core/actuation-base.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

namespace cito {
namespace python {
namespace bp = boost::python;

/**
 * @brief Expose ActuationModelFloatingBaseManipulationTpl to Python
 */
 
void exposeActuationFloatingBaseManipulation() {

    bp::register_ptr_to_python<boost::shared_ptr<ActuationModelFloatingBaseManipulation>>();

    typedef double Scalar;
    typedef crocoddyl::ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
    typedef crocoddyl::ActuationDataAbstractTpl<Scalar> ActuationData;

    // Register shared pointer for ActuationModelFloatingBaseManipulation

    // Expose the ActuationModelFloatingBaseManipulation class
    bp::class_<ActuationModelFloatingBaseManipulation,
               bp::bases<ActuationModelAbstract> >(
        "ActuationModelFloatingBaseManipulation",
        "Floating-base actuation model for manipulation tasks.",
        bp::init<boost::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> >,
                 bp::optional<std::string> >(
            bp::args("state", "floating_joint_name"),
            "Initialize the floating-base actuation model.\n\n"
            ":param state: State of a multibody system\n"
            ":param floating_joint_name: Name of the floating joint (default: 'cube_joint')")
    )
    .def("calc", &ActuationModelFloatingBaseManipulation::calc,
         bp::args("self", "data", "x", "u"),
         "Compute the actuation signal from joint-torque input.")
    .def("calcDiff", &ActuationModelFloatingBaseManipulation::calcDiff,
         bp::args("self", "data", "x", "u"),
         "Compute the derivatives of the actuation model.")
    .def("commands", &ActuationModelFloatingBaseManipulation::commands,
         bp::args("self", "data", "x", "tau"),
         "Extract the control input from the torque vector.")
    .def("torqueTransform", &ActuationModelFloatingBaseManipulation::torqueTransform,
         bp::args("self", "data", "x", "u"),
         "Compute the torque transform matrix.")
    .def("createData", &ActuationModelFloatingBaseManipulation::createData,
         bp::args("self"),
         "Create the actuation data.");

}

}  // namespace python
}  // namespace cito

