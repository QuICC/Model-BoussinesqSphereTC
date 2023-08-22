/**
 * @file MomentumKernel.hpp
 * @brief Physical kernel for the Momentum nonlinear kernel
 */

#ifndef QUICC_PHYSICAL_MOMENTUMKERNEL_HPP
#define QUICC_PHYSICAL_MOMENTUMKERNEL_HPP

// System includes
//
#include <memory>

// Project includes
//
#include "QuICC/PhysicalKernels/IPhysicalKernel.hpp"

namespace QuICC {

namespace Physical {

namespace Kernel {

   /**
    * @brief Physical kernel for the Momentum nonlinear kernel
    */
   class MomentumKernel: public IPhysicalKernel
   {
      public:
         /**
          * @brief Simple constructor
          */
         explicit MomentumKernel();

         /**
          * @brief Simple empty destructor
          */
         virtual ~MomentumKernel() = default;

         /**
          * @brief Set the physical mesh on which kernel is working
          */
         virtual void setMesh(std::shared_ptr<std::vector<Array> > spMesh) override;

         /**
          * @brief Set the smart pointer to the temperature field
          *
          * \param name Name of the field
          * \param spField Shared pointer to the scalar field
          */
         virtual void setTemperature(std::size_t name, Framework::Selector::VariantSharedScalarVariable spField);

         /**
          * @brief Set the smart pointer to the velocitt field
          *
          * \param name Name of the field
          * \param spField Shared pointer to the vector field
          */
         virtual void setVelocity(std::size_t name, Framework::Selector::VariantSharedVectorVariable spField);

         /**
          * @brief Initialize kernel
          */
         void init(const MHDFloat inertia, const MHDFloat buoyancy);

         /**
          * @brief Compute the physical kernel
          *
          * @param rNLComp Nonlinear term component
          * @param id      ID of the component (allows for a more general implementation)
          */
         virtual void compute(Framework::Selector::PhysicalScalarField& rNLComp, FieldComponents::Physical::Id id) const override;

      protected:
         /**
          * @brief Get name ID of the unknown
          */
         std::size_t name() const;

      private:
         /**
          * @brief Name ID of the unknown
          */
         std::size_t mName;

         /**
          * @brief Name ID of the temperature field
          */
         std::size_t mTempName;

         /**
          * @brief Scaling constant for inertial term
          */
         MHDFloat mInertia;

         /**
          * @brief Scaling constant for Buoyancy term
          */
         MHDFloat mBuoyancy;

         /**
          * @brief Storage for the radial grid values (if required)
          */
         Array mRadius;

   };

   /// Typedef for a smart MomentumKernel
   typedef std::shared_ptr<MomentumKernel> SharedMomentumKernel;

}
}
}

#endif // QUICC_PHYSICAL_MOMENTUMKERNEL_HPP
