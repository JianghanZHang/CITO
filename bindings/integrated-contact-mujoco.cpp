#include "cito/python.hpp"
#include "cito/fwd.hpp"
#include "cito/actions/integrated-contact-mujoco.hpp"
#include "utils/copyable.hpp"
#include <crocoddyl/core/integ-action-base.hpp>
#include <crocoddyl/core/diff-action-base.hpp>


namespace cito {
namespace python {
namespace bp = boost::python;

void exposeIAMmujoco() {
    // Register shared pointer for IntegratedActionModelContactMj
    bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionModelContactMj>>();

    // Expose the IntegratedActionModelContactMj class
    bp::class_<IntegratedActionModelContactMj,
               bp::bases<crocoddyl::IntegratedActionModelAbstract> >(
                "IntegratedActionModelContactMj",
                "Integrated action model with MuJoCo-based contact dynamics.",
                bp::init<boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>,
                         bp::optional<double, bool> >(
                    bp::args("self", "differential", "time_step", "with_cost_residual"),
                    "Initialize the symplectic Euler integrator for contact dynamics.\n\n"
                    ":param differential: Differential action model\n"
                    ":param time_step: Integration step size\n"
                    ":param with_cost_residual: Flag indicating whether to compute cost residual"))
        // Binding for calc method that takes both state x and control u
        .def<void (IntegratedActionModelContactMj::*)(
            const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &IntegratedActionModelContactMj::calc,
            bp::args("self", "data", "x", "u"),
            "Integrate the differential action model using symplectic Euler scheme.\n\n"
            ":param data: Symplectic Euler data\n"
            ":param x: State vector\n"
            ":param u: Control input")
        // Binding for calc method for terminal states (without control u)
        .def<void (IntegratedActionModelContactMj::*)(
            const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &IntegratedActionModelContactMj::calc,
            bp::args("self", "data", "x"),
            "Integrate the total cost value for terminal nodes.\n\n"
            ":param data: Symplectic Euler data\n"
            ":param x: State vector")
        // Binding for calcDiff method that computes derivatives wrt x and u
        .def<void (IntegratedActionModelContactMj::*)(
            const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &IntegratedActionModelContactMj::calcDiff,
            bp::args("self", "data", "x", "u"),
            "Compute the partial derivatives of the symplectic Euler integrator.\n\n"
            ":param data: Symplectic Euler data\n"
            ":param x: State vector\n"
            ":param u: Control input")
        // Binding for calcDiff method for terminal nodes (without control u)
        .def<void (IntegratedActionModelContactMj::*)(
            const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &IntegratedActionModelContactMj::calcDiff,
            bp::args("self", "data", "x"),
            "Compute the partial derivatives of the cost for terminal nodes.\n\n"
            ":param data: Symplectic Euler data\n"
            ":param x: State vector")
        // Expose the createData method
        .def("createData", &IntegratedActionModelContactMj::createData,
             bp::args("self"),
             "Create the symplectic Euler action data.")
        // Expose the print method
        .def("print", &IntegratedActionModelContactMj::print,
             bp::args("self", "os"),
             "Print relevant information of the Euler integrator model.")
        // Expose A_ (Fx) with getter and setter
        .add_property(".",
                      bp::make_function(&IntegratedActionModelContactMj::get_A,
                                        bp::return_value_policy<bp::copy_const_reference>()),
                      bp::make_function(&IntegratedActionModelContactMj::set_A),
                      "Fx matrix (partial derivatives of next state with respect to the state).")
        // Expose B_ (Fu) with getter and setter
        .add_property(".",
                      bp::make_function(&IntegratedActionModelContactMj::get_B,
                                        bp::return_value_policy<bp::copy_const_reference>()),
                      bp::make_function(&IntegratedActionModelContactMj::set_B),
                      ".")
        .add_property("stateMappingDerivatives",
                      bp::make_function(&IntegratedActionModelContactMj::get_stateMappingDerivatives,
                                        bp::return_value_policy<bp::copy_const_reference>()),
                      bp::make_function(&IntegratedActionModelContactMj::set_stateMappingDerivatives),
                      ".")
        .add_property("stateMappingDerivativesNext",
                      bp::make_function(&IntegratedActionModelContactMj::get_stateMappingDerivativesNext,
                                        bp::return_value_policy<bp::copy_const_reference>()),
                      bp::make_function(&IntegratedActionModelContactMj::set_stateMappingDerivativesNext),
                      ".")
        .add_property("forces",
                      bp::make_function(&IntegratedActionModelContactMj::get_forces,
                                        bp::return_value_policy<bp::copy_const_reference>()),
                      bp::make_function(&IntegratedActionModelContactMj::set_forces),
                      ".");

    // Register the shared pointer type for IntegratedActionDataContactMj
    bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataContactMj>>();
    // Expose the IntegratedActionDataContactMj class
    bp::class_<IntegratedActionDataContactMj,
               bp::bases<crocoddyl::IntegratedActionDataAbstract> >(
        "IntegratedActionDataContactMj",
        "Symplectic Euler action data for MuJoCo-based contact dynamics.",
        bp::init<IntegratedActionModelContactMj*>(
            bp::args("self", "model"),
            "Create MuJoCo-based contact dynamics symplectic Euler action data.\n\n"
            ":param model: MuJoCo-based contact dynamics action model"))
        // Expose the differential property
        .add_property("differential",
                      bp::make_getter(&IntegratedActionDataContactMj::differential,
                                      bp::return_internal_reference<>()),
                      "Differential model data.")
        // Expose the contact property
        .add_property("contact",
                      bp::make_getter(&IntegratedActionDataContactMj::contact,
                                      bp::return_internal_reference<>()),
                      "Contact data.");
}
}  // namespace python
}  // namespace cito
