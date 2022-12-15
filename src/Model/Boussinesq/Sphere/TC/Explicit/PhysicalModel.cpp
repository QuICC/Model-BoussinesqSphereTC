/**
 * @file PhysicalModel.cpp
 * @brief Source of the Boussinesq thermal convection in a sphere (Toroidal/Poloidal formulation) without coupled solve (standard implementation)
 */

// System includes
//

// Class include
// Project includes
#include "QuICC/Model/Boussinesq/Sphere/TC/Explicit/PhysicalModel.hpp"
#include "QuICC/Model/Boussinesq/Sphere/TC/Explicit/ModelBackend.hpp"
#include "QuICC/Model/PyModelBackend.hpp"

// Force use of C++ backend
#define QUICC_MODEL_BOUSSINESQSPHERETC_EXPLICIT_BACKEND_CPP

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

namespace Explicit {

   std::string PhysicalModel::PYMODULE()
   {
      return "boussinesq.sphere.tc.explicit.physical_model";
   }

   void PhysicalModel::init()
   {
#ifdef QUICC_MODEL_BOUSSINESQSPHERETC_EXPLICIT_BACKEND_CPP
      IPhysicalModel<Simulation,StateGenerator,VisualizationGenerator>::init();

      this->mpBackend = std::make_shared<ModelBackend>();
#else
      IPhysicalPyModel<Simulation,StateGenerator,VisualizationGenerator>::init();

      this->mpBackend = std::make_shared<PyModelBackend>(this->PYMODULE(), this->PYCLASS());
#endif
   }

}
}
}
}
}
}
