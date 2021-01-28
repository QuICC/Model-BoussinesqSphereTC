/**
 * @file PhysicalModel.cpp
 * @brief Source of the Boussinesq thermal convection in a sphere (Toroidal/Poloidal formulation) without coupled solve (standard implementation)
 */

// Configuration includes
//

// System includes
//

// External includes
//

// Class include
//
#include "QuICC/Model/Boussinesq/Sphere/TC/Explicit/PhysicalModel.hpp"

// Project includes
//

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

}
}
}
}
}
}
