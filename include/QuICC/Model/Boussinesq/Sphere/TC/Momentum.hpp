/**
 * @file Momentum.hpp
 * @brief Implementation of the vector Navier-Stokes equation for the Boussinesq thermal convection sphere
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

#ifndef QUICC_MODEL_BOUSSINESQ_SPHERE_TC_MOMENTUM_HPP
#define QUICC_MODEL_BOUSSINESQ_SPHERE_TC_MOMENTUM_HPP

// Configuration includes
//

// System includes
//

// External includes
//

// Project includes
//
#include "QuICC/Base/Typedefs.hpp"
#include "QuICC/TypeSelectors/ScalarSelector.hpp"
#include "QuICC/Equations/IVectorEquation.hpp"

namespace QuICC {

namespace Equations {

namespace Boussinesq {

namespace Sphere {

namespace TC {

   /**
    * @brief Implementation of the vector Navier-Stokes equation for the Boussinesq thermal convection in a sphere
    */
   class Momentum: public IVectorEquation
   {
      public:
         /**
          * @brief Simple constructor
          *
          * @param spEqParams  Shared equation parameters
          */
         Momentum(SharedEquationParameters spEqParams);

         /**
          * @brief Simple empty destructor
          */
         virtual ~Momentum();

         /**
          * @brief Compute the nonlinear interaction term
          *
          * @param rNLComp Nonlinear term component
          * @param id      ID of the component (allows for a more general implementation)
          */
         virtual void computeNonlinear(Datatypes::PhysicalScalarType& rNLComp, FieldComponents::Physical::Id id) const;

      protected:
         /**
          * @brief Set variable requirements
          */
         virtual void setRequirements();

         /**
          * @brief Set the equation coupling information
          */
         virtual void setCoupling();

         /**
          * @brief Set the nonlinear integration components
          */
         virtual void setNLComponents();

      private:
   };

}
}
}
}
}

#endif // QUICC_MODEL_BOUSSINESQ_SPHERE_TC_MOMENTUM_HPP