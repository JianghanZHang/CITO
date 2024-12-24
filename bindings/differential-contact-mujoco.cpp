// // #include "cito/python.hpp"
// // #include "cito/fwd.hpp"
// // #include "cito/actions/differential-contact-mujoco.hpp"
// // #include "utils/copyable.hpp"
// // #include <crocoddyl/core/diff-action-base.hpp>
// // #include <pybind11/pybind11.h>
// // #include "cito/utils/pyboost11-converter.hpp"
// // #include <mujoco/mujoco.h>
// // #include <mujoco/mjdata.h>
// // #include <mujoco/mjmodel.h>

// // namespace cito {

// // namespace python {
// // namespace bp = boost::python;
// // namespace py = pybind11;

// // void exposeDAMmujoco() {

// //     pyboost11::converter<mjModel>();  // mjModelWrapper (py::object)
// //     pyboost11::converter<mjData>();  // mjDataWrapper (py::object)

// //     bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionModelContactMj>>();

// //     bp::class_<DifferentialActionModelContactMj,
// //                bp::bases<crocoddyl::DifferentialActionModelAbstract> >(
// //         "DifferentialActionModelContactMj",
// //         "This is a dummy class (only for data storage) to maintain Crocoddyl's intended class hierarchy\n",
// //         bp::init<const py::object,  // Now expects pybind11::object for mjModelWrapper
// //                  py::object,  // Expects pybind11::object for mjDataWrapper
// //                  boost::shared_ptr<crocoddyl::StateMultibody>,
// //                  boost::shared_ptr<crocoddyl::ActuationModelAbstract>,
// //                  boost::shared_ptr<crocoddyl::CostModelSum>,
// //                  bp::optional<boost::shared_ptr<crocoddyl::ConstraintModelManager>>>(
// //             bp::args("self", "mjModelWrapper", "mjDataWrapper", "state", "actuation", "costs", "constraints"),
// //             "Initialize the MuJoCo-based contact dynamics action model.\n\n"
// //             ":param mjModel: MuJoCo model (pybind11::object)\n"
// //             ":param mjData: MuJoCo data (pybind11::object)\n"
// //             ":param state: multibody state\n"
// //             ":param actuation: abstract actuation model\n"
// //             ":param costs: stack of cost functions\n"
// //             ":param constraints: optional stack of constraint functions"))
// //         .def<void (DifferentialActionModelContactMj::*)(
// //             const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
// //             const Eigen::Ref<const Eigen::VectorXd>&,
// //             const Eigen::Ref<const Eigen::VectorXd>&)>(
// //             "calc", &DifferentialActionModelContactMj::calc,
// //             bp::args("self", "data", "x", "u"),
// //             "Compute the next state and cost value with MuJoCo-based contact dynamics.\n\n"
// //             ":param data: action data for MuJoCo-based contact dynamics\n"
// //             ":param x: time-continuous state vector\n"
// //             ":param u: time-continuous control input")
// //         .def<void (DifferentialActionModelContactMj::*)(
// //             const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
// //             const Eigen::Ref<const Eigen::VectorXd>&)>(
// //             "calc", &DifferentialActionModelContactMj::calc,
// //             bp::args("self", "data", "x"),
// //             "Compute the next state and cost value for terminal nodes.\n\n"
// //             ":param data: action data\n"
// //             ":param x: time-continuous state vector")
// //         .def<void (DifferentialActionModelContactMj::*)(
// //             const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
// //             const Eigen::Ref<const Eigen::VectorXd>&,
// //             const Eigen::Ref<const Eigen::VectorXd>&)>(
// //             "calcDiff", &DifferentialActionModelContactMj::calcDiff,
// //             bp::args("self", "data", "x", "u"),
// //             "Compute the partial derivatives of the MuJoCo-based contact dynamics and costs.\n\n"
// //             ":param data: action data\n"
// //             ":param x: time-continuous state vector\n"
// //             ":param u: time-continuous control input")
// //         .def<void (DifferentialActionModelContactMj::*)(
// //             const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
// //             const Eigen::Ref<const Eigen::VectorXd>&)>(
// //             "calcDiff", &DifferentialActionModelContactMj::calcDiff,
// //             bp::args("self", "data", "x"),
// //             "Compute the partial derivatives of the system and costs for terminal nodes.\n\n"
// //             ":param data: action data\n"
// //             ":param x: time-continuous state vector")
// //         .def("createData", &DifferentialActionModelContactMj::createData,
// //              bp::args("self"),
// //              "Create the MuJoCo-based differential action data.");
// // }
// // }  // namespace python
// // }  // namespace cito


#include "cito/python.hpp"
#include "cito/fwd.hpp"
#include "cito/actions/differential-contact-mujoco.hpp"
#include "utils/copyable.hpp"
#include <crocoddyl/core/diff-action-base.hpp>
#include "cito/actions/differential-contact-mujoco.hpp"
#include "utils/copyable.hpp"
#include <crocoddyl/core/diff-action-base.hpp>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>

namespace cito{

namespace python{
namespace bp = boost::python;
void exposeDAMmujoco(){

    bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionModelContactMj>>();

    bp::class_<DifferentialActionModelContactMj,
               bp::bases<crocoddyl::DifferentialActionModelAbstract> >(
                "DifferentialActionModelContactMj",
                "This is a dummy class (only for data storage) to maintain Crocoddyl's intended class hierarchy\n",
        bp::init<const bp::object,  // Now expects pybind11::object for mjModelWrapper
                 boost::shared_ptr<crocoddyl::StateMultibody>,
                 boost::shared_ptr<crocoddyl::ActuationModelAbstract>,
                 boost::shared_ptr<crocoddyl::CostModelSum>,
                 bp::optional<boost::shared_ptr<crocoddyl::ConstraintModelManager>>>(
            bp::args("self", "mjModel", "state", "actuation", "costs", "constraints"),
            "Initialize the MuJoCo-based contact dynamics action model.\n\n"
            ":param mjModel: const MuJoCo model\n"  // Indicate that mjModel is const
            ":param state: multibody state\n"
            ":param actuation: abstract actuation model\n"
            ":param costs: stack of cost functions\n"
            ":param constraints: optional stack of constraint functions"))
        // Binding for calc method that takes both state x and control u
        .def<void (DifferentialActionModelContactMj::*)(
            const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &DifferentialActionModelContactMj::calc,
            bp::args("self", "data", "x", "u"),
            "Compute the next state and cost value with MuJoCo-based contact dynamics.\n\n"
            ":param data: action data for MuJoCo-based contact dynamics\n"
            ":param x: time-continuous state vector\n"
            ":param u: time-continuous control input")
        // Binding for calc method for terminal states (without control u)
        .def<void (DifferentialActionModelContactMj::*)(
            const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> &,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &DifferentialActionModelContactMj::calc,
            bp::args("self", "data", "x"),
            "Compute the next state and cost value for terminal nodes.\n\n"
            ":param data: action data\n"
            ":param x: time-continuous state vector")
        // Binding for calcDiff method that computes derivatives wrt x and u
        .def<void (DifferentialActionModelContactMj::*)(
            const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> &,
            const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &DifferentialActionModelContactMj::calcDiff,
            bp::args("self", "data", "x", "u"),
            "Compute the partial derivatives of the MuJoCo-based contact dynamics and costs.\n\n"
            ":param data: action data\n"
            ":param x: time-continuous state vector\n"
            ":param u: time-continuous control input")
        // Binding for calcDiff method for terminal nodes (without control u)
        .def<void (DifferentialActionModelContactMj::*)(
            const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> &,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &DifferentialActionModelContactMj::calcDiff,
            bp::args("self", "data", "x"),
            "Compute the partial derivatives of the system and costs for terminal nodes.\n\n"
            ":param data: action data\n"
            ":param x: time-continuous state vector")
        // Expose the createData method
        .def("createData", &DifferentialActionModelContactMj::createData,
             bp::args("self"),
             "Create the MuJoCo-based differential action data.")
        // Expose CopyableVisitor for class copying support
        .def(CopyableVisitor<DifferentialActionModelContactMj>());

    // Register the shared pointer type for DifferentialActionDataContactMjTpl
    bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataContactMj>>();

    // Expose the DifferentialActionDataContactMjTpl class
    bp::class_<DifferentialActionDataContactMj,
               bp::bases<crocoddyl::DifferentialActionDataAbstract> >(
        "DifferentialActionDataContactMj",
        "Differential action data for MuJoCo-based contact dynamics.",
        bp::init<DifferentialActionModelContactMj*>(
            bp::args("self", "model"),
            "Create MuJoCo-based contact dynamics differential action data.\n\n"
            ":param model: MuJoCo-based contact dynamics action model"))
        // Expose the pinocchio property (if required, based on internal model)
        .add_property("pinocchio",
                      bp::make_getter(&DifferentialActionDataContactMj::pinocchio,
                                      bp::return_internal_reference<>()),
                      "Pinocchio data for the multibody system.")
        // Expose the costs property
        .add_property("costs",
                      bp::make_getter(&DifferentialActionDataContactMj::costs,
                                      bp::return_value_policy<bp::return_by_value>()),
                      "Total cost data.")
        // Expose other properties as needed
        .def(CopyableVisitor<DifferentialActionDataContactMj>());
}

}
}  // namespace cito
