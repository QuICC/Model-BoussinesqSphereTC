/**
 * @file MomentumKernel.cpp
 * @brief Source of physical space kernel for the Momentum equation
 */

// Configuration includes
//

// System includes
//

// External includes
//

// Class include
//
#include "QuICC/Model/Boussinesq/Sphere/TC/MomentumKernel.hpp"

// Project includes
//
#include "QuICC/PhysicalOperators/Cross.hpp"

namespace QuICC {

namespace Physical {

namespace Kernel {

   MomentumKernel::MomentumKernel()
      : IPhysicalKernel()
   {
   }

   MomentumKernel::~MomentumKernel()
   {
   }

   std::size_t MomentumKernel::name() const
   {
      return this->mName;
   }

   void MomentumKernel::setVelocity(std::size_t name, Framework::Selector::VariantSharedVectorVariable spField)
   {
      // Safety assertion
      assert(this->mScalars.count(name) + this->mVectors.count(name) == 0);

      this->mName = name;

      this->setField(name, spField);
   }

   void MomentumKernel::init(const MHDFloat inertia)
   {
      // Set scaling constants
      this->mInertia = inertia;
   }

   void MomentumKernel::compute(Framework::Selector::PhysicalScalarField& rNLComp, FieldComponents::Physical::Id id) const
   {
      ///
      /// Compute \f$\left(\nabla\wedge\vec u\right)\wedge\vec u\f$
      ///
      switch(id)
      {
         case(FieldComponents::Physical::R):
            std::visit([&](auto&& v){Physical::Cross<FieldComponents::Physical::THETA,FieldComponents::Physical::PHI>::set(rNLComp, v->dom(0).curl(), v->dom(0).phys(), this->mInertia);}, this->vector(this->name()));
            break;
         case(FieldComponents::Physical::THETA):
            std::visit([&](auto&& v){Physical::Cross<FieldComponents::Physical::PHI,FieldComponents::Physical::R>::set(rNLComp, v->dom(0).curl(), v->dom(0).phys(), this->mInertia);}, this->vector(this->name()));
            break;
         case(FieldComponents::Physical::PHI):
            std::visit([&](auto&& v){Physical::Cross<FieldComponents::Physical::R,FieldComponents::Physical::THETA>::set(rNLComp, v->dom(0).curl(), v->dom(0).phys(), this->mInertia);}, this->vector(this->name()));
            break;
         default:
            assert(false);
            break;
      }
   }

}
}
}
