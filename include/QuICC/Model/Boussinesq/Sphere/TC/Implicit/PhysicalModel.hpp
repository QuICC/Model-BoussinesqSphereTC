/**
 * @file PhysicalModel.hpp
 * @brief Implementation of the Boussinesq thermal convection in a sphere (Toroidal/Poloidal formulation)
 */

#ifndef QUICC_MODEL_BOUSSINESQ_SPHERE_TC_IMPLICIT_PHYSICALMODEL_HPP
#define QUICC_MODEL_BOUSSINESQ_SPHERE_TC_IMPLICIT_PHYSICALMODEL_HPP

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

}
}
}
}
}
}

//
// Block compilation of unusable parallelisation algorithms
//
#ifdef QUICC_MPIALGO_SINGLE1D
#error "The SINGLE1D parallelisation is not supported!"
#endif //QUICC_MPIALGO_SINGLE1D
#if defined QUICC_MPIALGO_TUBULAR && !defined QUICC_SPLINALG_MUMPS && !defined QUICC_MPISPSOLVE
#error "The TUBULAR parallelisation is not supported!"
#endif //defined QUICC_MPIALGO_TUBULAR && !defined QUICC_SPLINALG_MUMPS && !defined QUICC_MPISPSOLVE

#endif // QUICC_MODEL_BOUSSINESQ_SPHERE_TC_IMPLICIT_PHYSICALMODEL_HPP
