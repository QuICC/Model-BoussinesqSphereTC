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
#include "QuICC/SpatialScheme/ISpatialScheme.hpp"
#include "QuICC/PhysicalOperators/Cross.hpp"
#include "QuICC/PhysicalOperators/SphericalBuoyancy.hpp"

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

   void MomentumKernel::setTemperature(std::size_t name, Framework::Selector::VariantSharedScalarVariable spField)
   {
      // Safety assertion
      assert(this->mScalars.count(name) + this->mVectors.count(name) == 0);

      this->mTempName = name;

      this->setField(name, spField);
   }

   void MomentumKernel::init(const MHDFloat inertia, const MHDFloat buoyancy)
   {
      // Set scaling constants
      this->mInertia = inertia;
      this->mBuoyancy = buoyancy;
   }

   void MomentumKernel::setMesh(std::shared_ptr<std::vector<Array> > spMesh)
   {
      IPhysicalKernel::setMesh(spMesh);

      this->mRadius = this->mspMesh->at(0);
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

      // Buoyancy
      std::visit([&](auto&& s){Physical::SphericalBuoyancy::sub(rNLComp, id, s->dom(0).res(), this->mRadius, s->dom(0).phys(), this->mBuoyancy);}, this->scalar(this->mTempName));
   }

}
}
}
