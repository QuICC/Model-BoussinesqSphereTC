/**
 * @file Transport.hpp
 * @brief Implementation of the transport equation for the Boussinesq thermal convection in a sphere
 */

#ifndef QUICC_MODEL_BOUSSINESQ_SPHERE_TC_TRANSPORT_HPP
#define QUICC_MODEL_BOUSSINESQ_SPHERE_TC_TRANSPORT_HPP

// System includes
//
#include <memory>

// Project includes
//
#include "QuICC/Equations/IScalarEquation.hpp"

namespace QuICC {

namespace Equations {

namespace Boussinesq {

namespace Sphere {

namespace TC {

   /**
    * @brief Implementation of the transport equation for the Boussinesq thermal convection in a sphere
    */
   class Transport: public IScalarEquation
   {
      public:
         /**
          * @brief Simple constructor
          *
          * @param spEqParams  Shared equation parameters
          */
         Transport(SharedEquationParameters spEqParams, SpatialScheme::SharedCISpatialScheme spScheme, std::shared_ptr<Model::IModelBackend> spBackend);

         /**
          * @brief Simple empty destructor
          */
         virtual ~Transport() = default;

         /**
          * @brief Initialize nonlinear interaction kernel
          */
         void initNLKernel(const bool force = false) final;

      protected:
         /**
          * @brief Set nonlinear component path
          */
         void setNLComponents() final;

         /**
          * @brief Set variable requirements
          */
         void setRequirements() final;

         /**
          * @brief Set the equation coupling information
          */
         void setCoupling() final;

      private:
   };

}
}
}
}
}

#endif // QUICC_MODEL_BOUSSINESQ_SPHERE_TC_TRANSPORT_HPP
