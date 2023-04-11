/**
 * @file PhysicalModel.cpp
 * @brief Source of the Boussinesq thermal convection in a sphere (Toroidal/Poloidal formulation)
 */

// System includes
//

// Project includes
//
#include "QuICC/Model/Boussinesq/Sphere/TC/Implicit/PhysicalModel.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

namespace Implicit {

   std::string PhysicalModel::PYMODULE()
   {
      return "boussinesq.sphere.tc.implicit.physical_model";
   }

} // Implicit
} // TC
} // Sphere
} // Boussinesq
} // Model
} // QuICC
