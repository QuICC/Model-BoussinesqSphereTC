/**
 * @file PhysicalModel.hpp
 * @brief Implementation of the Boussinesq thermal convection in a sphere (Toroidal/Poloidal formulation)
 */

#ifndef QUICC_MODEL_BOUSSINESQ_SPHERE_TC_IMPLICIT_PHYSICALMODEL_HPP
#define QUICC_MODEL_BOUSSINESQ_SPHERE_TC_IMPLICIT_PHYSICALMODEL_HPP

// System includes
//
#include <string>

// Project includes
//
#include "Model/Boussinesq/Sphere/TC/ITCModel.hpp"
#include "QuICC/SpatialScheme/3D/WLFm.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

namespace Implicit {

   /**
    * @brief Implementation of the Boussinesq thermal convection sphere model (Toroidal/Poloidal formulation)
    */
   class PhysicalModel: public ITCModel
   {
      public:
         /// Typedef for the spatial scheme used
         typedef SpatialScheme::WLFm SchemeType;

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

      protected:

      private:
   };

} // Implicit
} // TC
} // Sphere
} // Boussines
} // Model
} // QuICC

#endif // QUICC_MODEL_BOUSSINESQ_SPHERE_TC_IMPLICIT_PHYSICALMODEL_HPP
