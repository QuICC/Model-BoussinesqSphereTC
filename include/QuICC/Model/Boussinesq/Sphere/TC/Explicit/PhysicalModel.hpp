/**
 * @file PhysicalModel.hpp
 * @brief Implementation of the Boussinesq thermal convection in a sphere (Toroidal/Poloidal formulation) without coupled solve (standard implementation)
 */

#ifndef QUICC_MODEL_BOUSSINESQ_SPHERE_TC_EXPLICIT_PHYSICALMODEL_HPP
#define QUICC_MODEL_BOUSSINESQ_SPHERE_TC_EXPLICIT_PHYSICALMODEL_HPP

// Model version
#define QUICC_VERSION_MODEL_MAJOR 1
#define QUICC_VERSION_MODEL_MINOR 0
#define QUICC_VERSION_MODEL_PATCH 0

// Configuration includes
//

// System includes
//
#include <string>

// External includes
//

// Project includes
//
#include "QuICC/Model/Boussinesq/Sphere/TC/ITCModel.hpp"
#include "QuICC/SpatialScheme/3D/WLFl.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

namespace Explicit {

   /**
    * @brief Implementation of the Boussinesq thermal convection sphere model (Toroidal/Poloidal formulation) without coupled solve (standard implementation)
    */
   class PhysicalModel: public ITCModel
   {
      public:
         /// Typedef for the spatial scheme used
         typedef SpatialScheme::WLFl SchemeType;

         /**
          * @brief Constructor
          */
         PhysicalModel() = default;

         /**
          * @brief Destructor
          */
         virtual ~PhysicalModel() = default;

         /// Python script/module name
         virtual std::string PYMODULE() override;

         /**
          * @brief Initialize specialized backend
          */
         virtual void init() override;

      protected:

      private:
   };

}
}
}
}
}
}

#endif // QUICC_MODEL_BOUSSINESQ_SPHERE_TC_EXPLICIT_PHYSICALMODEL_HPP
