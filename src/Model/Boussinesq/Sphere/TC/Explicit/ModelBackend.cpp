/** 
 * @file ModelBackend.cpp
 * @brief Source of the interface for model backend
 */

// System includes
//
#include <stdexcept>

// Project includes
//
#include "QuICC/Model/Boussinesq/Sphere/TC/Explicit/ModelBackend.hpp"
#include "QuICC/ModelOperator/Time.hpp"
#include "QuICC/ModelOperator/ImplicitLinear.hpp"
#include "QuICC/ModelOperator/ExplicitLinear.hpp"
#include "QuICC/ModelOperator/ExplicitNonlinear.hpp"
#include "QuICC/ModelOperator/ExplicitNextstep.hpp"
#include "QuICC/ModelOperator/Stencil.hpp"
#include "QuICC/ModelOperator/Boundary.hpp"
#include "QuICC/ModelOperatorBoundary/FieldToRhs.hpp"
#include "QuICC/ModelOperatorBoundary/SolverHasBc.hpp"
#include "QuICC/ModelOperatorBoundary/SolverNoTau.hpp"
#include "QuICC/ModelOperatorBoundary/Stencil.hpp"
#include "QuICC/Enums/FieldIds.hpp"
#include "QuICC/PhysicalNames/Coordinator.hpp"
#include "QuICC/PhysicalNames/Velocity.hpp"
#include "QuICC/PhysicalNames/Temperature.hpp"
#include "QuICC/NonDimensional/Prandtl.hpp"
#include "QuICC/Tools/IdToHuman.hpp"
#include "QuICC/Resolutions/Tools/IndexCounter.hpp"
#include "QuICC/SparseSM/Worland/Id.hpp"
#include "QuICC/SparseSM/Worland/I2.hpp"
#include "QuICC/SparseSM/Worland/I4.hpp"
#include "QuICC/SparseSM/Worland/I2Lapl.hpp"
#include "QuICC/SparseSM/Worland/I4Lapl.hpp"
#include "QuICC/SparseSM/Worland/I4Lapl2.hpp"
#include "QuICC/SparseSM/Worland/Boundary/Value.hpp"
#include "QuICC/SparseSM/Worland/Boundary/D1.hpp"
#include "QuICC/SparseSM/Worland/Boundary/D2.hpp"
#include "QuICC/SparseSM/Worland/Boundary/R1D1DivR1.hpp"
#include "QuICC/SparseSM/Worland/Boundary/Operator.hpp"
#include "QuICC/SparseSM/Worland/Stencil/Value.hpp"
#include "QuICC/SparseSM/Worland/Stencil/D1.hpp"
#include "QuICC/SparseSM/Worland/Stencil/R1D1DivR1.hpp"
#include "QuICC/SparseSM/Worland/Stencil/ValueD1.hpp"
#include "QuICC/SparseSM/Worland/Stencil/ValueD2.hpp"

#include "QuICC/Polynomial/Worland/WorlandBase.hpp"
#include "QuICC/Equations/CouplingIndexType.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

namespace Explicit {

   ModelBackend::ModelBackend(const std::string pyModule, const std::string pyClass)
      : PyModelBackend(pyModule, pyClass), mUseGalerkin(false)
   {
   }

   std::vector<std::string> ModelBackend::fieldNames() const
   {
      std::vector<std::string> names = {"velocity", "temperature"};

      return names;
   }

   std::vector<std::string> ModelBackend::paramNames() const
   {
      std::vector<std::string> names = {"prandtl", "rayleigh"};

      return names;
   }

   std::vector<bool> ModelBackend::isPeriodicBox() const
   {
      std::vector<bool> periodic = {false, false, false};

      return periodic;
   }

   void ModelBackend::enableGalerkin(const bool flag)
   {
      this->mUseGalerkin = flag;
      this->mpWrapper->enableGalerkin(flag);
   }

   std::map<std::string,MHDFloat> ModelBackend::automaticParameters(const std::map<std::string,MHDFloat>& cfg) const
   {
      std::map<std::string,MHDFloat> params;

      return params;
   }

   ModelBackend::SpectralFieldIds ModelBackend::implicitFields(const SpectralFieldId& fId) const
   {
      SpectralFieldIds fields = {fId};

      return fields;
   }

   void ModelBackend::equationInfo(bool& isComplex, SpectralFieldIds& im, SpectralFieldIds& exL, SpectralFieldIds& exNL, SpectralFieldIds& exNS, int& indexMode, const SpectralFieldId& fId, const Resolution& res) const
   {
      // Operators are real
      isComplex = false;

      // Implicit coupled fields
      im = this->implicitFields(fId);

      // Explicit linear terms
      exL.clear();

      // Explicit nonlinear terms
      exNL.clear();

      // Explicit nextstep terms
      exNS.clear();

      // Index mode
      indexMode = static_cast<int>(Equations::CouplingIndexType::SLOWEST_MULTI_RHS);
   }

   void ModelBackend::blockSize(int& tN, int& gN, ArrayI& shift, int& rhs, const SpectralFieldId& fId, const Resolution& res, const std::vector<MHDFloat>& eigs, const BcMap& bcs) const
   {
      auto nN = res.counter().dimensions(Dimensions::Space::SPECTRAL, eigs.at(0))(0);
      tN = nN;

      int shiftR;
      if(this->mUseGalerkin)
      {
         if(fId == std::make_pair(PhysicalNames::Velocity::id(), FieldComponents::Spectral::TOR) ||
               fId == std::make_pair(PhysicalNames::Temperature::id(), FieldComponents::Spectral::SCALAR))
         {
            shiftR = 1;
         }
         else if(fId == std::make_pair(PhysicalNames::Velocity::id(), FieldComponents::Spectral::POL))
         {
            shiftR = 2;
         }
         else
         {
            shiftR = 0;
         }

         gN = (nN - shiftR);
      }
      else
      {
         shiftR = 0;
         gN = nN;
      }

      // Set galerkin shifts
      shift(0) = shiftR;
      shift(1) = 0;
      shift(2) = 0;

      rhs = 1;
   }

   void ModelBackend::operatorInfo(ArrayI& tauN, ArrayI& galN, MatrixI& galShift, ArrayI& rhsCols, ArrayI& sysN, const SpectralFieldId& fId, const Resolution& res, const Equations::Tools::ICoupling& coupling, const BcMap& bcs) const
   {
      // Loop overall matrices/eigs 
      for(int idx = 0; idx < tauN.size(); ++idx)
      {
         auto eigs = coupling.getIndexes(res, idx);

         int tN, gN, rhs;
         ArrayI shift(3);

         this->blockSize(tN, gN, shift, rhs, fId, res, eigs, bcs);

         tauN(idx) = tN;
         galN(idx) = gN;
         galShift.row(idx) = shift;
         rhsCols(idx) = rhs;

         // Compute system size
         int sN = 0;
         for(auto f: this->implicitFields(fId))
         {
            this->blockSize(tN, gN, shift, rhs, f, res, eigs, bcs);
            sN += gN;
         }

         if(sN == 0)
         {
            sN = galN(idx);
         }

         sysN(idx) = sN;
      }
   }

   void ModelBackend::modelMatrix(DecoupledZSparse& rModelMatrix, const std::size_t opId, const Equations::CouplingInformation::FieldId_range imRange, const int matIdx, const std::size_t bcType, const Resolution& res, const std::vector<MHDFloat>& eigs, const BcMap& bcs, const NonDimensional::NdMap& nds) const
   {
      assert(eigs.size() == 1);
      int l = eigs.at(0);

      assert(std::distance(imRange.first, imRange.second) == 1);
      auto fieldId = imRange.first->first;
      auto compId = imRange.first->second;

      auto nN = res.counter().dimensions(Dimensions::Space::SPECTRAL, eigs.at(0))(0);

      auto a = Polynomial::Worland::WorlandBase::ALPHA_CHEBYSHEV;
      auto b = Polynomial::Worland::WorlandBase::DBETA_CHEBYSHEV;

      // Time operator
      if(opId == ModelOperator::Time::id())
      {
         if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::TOR)
         {
            SparseSM::Worland::I2 spasm(nN, nN, a, b, l);
            rModelMatrix.real() = spasm.mat();
         }
         else if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::POL)
         {
            SparseSM::Worland::I4Lapl spasm(nN, nN, a, b, l);
            rModelMatrix.real() = spasm.mat();
         }
         else if(fieldId == PhysicalNames::Temperature::id() && compId == FieldComponents::Spectral::SCALAR)
         {
            auto Pr = nds.find(NonDimensional::Prandtl::id())->second->value();
            SparseSM::Worland::I2 spasm(nN, nN, a, b, l);
            rModelMatrix.real() = Pr*spasm.mat();
         }
      }
      // Linear operator
      else if(opId == ModelOperator::ImplicitLinear::id())
      {
         if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::TOR)
         {
            SparseSM::Worland::I2Lapl spasm(nN, nN, a, b, l);
            rModelMatrix.real() = spasm.mat();
         }
         else if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::POL)
         {
            SparseSM::Worland::I4Lapl2 spasm(nN, nN, a, b, l);
            rModelMatrix.real() = spasm.mat();
         }
         else if(fieldId == PhysicalNames::Temperature::id() && compId == FieldComponents::Spectral::SCALAR)
         {
            SparseSM::Worland::I2Lapl spasm(nN, nN, a, b, l);
            rModelMatrix.real() = spasm.mat();
         }
      }
      // Boundary operator
      else if(opId == ModelOperator::Boundary::id())
      {
         rModelMatrix.real().resize(nN, nN);
      }
      else
      {
         throw std::logic_error("Requested operator type is not implemented");
      }

      if((this->mUseGalerkin && bcType == ModelOperatorBoundary::SolverNoTau::id()) || (this->mUseGalerkin && opId == ModelOperator::Boundary::id()))
      {
         auto bcId = bcs.find(PhysicalNames::Coordinator::tag(fieldId))->second;
         auto stencil = rModelMatrix.real(); 

         int s = 0;
         if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::TOR)
         {
            s = 1;
            if(bcId == 0)
            {
               SparseSM::Worland::Stencil::Value bc(nN, nN-1, a, b, l);
               stencil = bc.mat(); 
            }
            else if(bcId == 1)
            {
               SparseSM::Worland::Stencil::R1D1DivR1 bc(nN, nN-1, a, b, l);
               stencil = bc.mat(); 
            }
            else
            {
               throw std::logic_error("Boundary conditions for Temperature not implemented");
            }
         }
         else if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::POL)
         {
            s = 2;
            if(bcId == 0)
            {
               SparseSM::Worland::Stencil::ValueD1 bc(nN, nN-2, a, b, l);
               stencil = bc.mat(); 
            }
            else if(bcId == 1)
            {
               SparseSM::Worland::Stencil::ValueD2 bc(nN, nN-2, a, b, l);
               stencil = bc.mat(); 
            }
            else
            {
               throw std::logic_error("Boundary conditions for Temperature not implemented");
            }
         }
         else if(fieldId == PhysicalNames::Temperature::id() && compId == FieldComponents::Spectral::SCALAR)
         {
            s = 1;
            if(bcId == 0)
            {
               SparseSM::Worland::Stencil::Value bc(nN, nN-1, a, b, l);
               stencil = bc.mat(); 
            }
            else if(bcId == 1)
            {
               SparseSM::Worland::Stencil::D1 bc(nN, nN-1, a, b, l);
               stencil = bc.mat(); 
            }
            else
            {
               throw std::logic_error("Boundary conditions for Temperature not implemented");
            }
         }

         SparseSM::Worland::Id qId(nN-s, nN, a, b, l, 0, s);
         rModelMatrix.real() = qId.mat()*(rModelMatrix.real()*stencil);
      }
      else if(bcType == ModelOperatorBoundary::SolverHasBc::id())
      {
         auto bcId = bcs.find(PhysicalNames::Coordinator::tag(fieldId))->second;

         SparseSM::Worland::Boundary::Operator bcOp(nN, nN, a, b, l);

         if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::TOR)
         {
            if(bcId == 0)
            {
               bcOp.addRow<SparseSM::Worland::Boundary::Value>();
            }
            else if(bcId == 1)
            {
               bcOp.addRow<SparseSM::Worland::Boundary::R1D1DivR1>();
            }
            else
            {
               throw std::logic_error("Boundary conditions for Temperature not implemented");
            }
         }
         else if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::POL)
         {
            if(bcId == 0)
            {
               bcOp.addRow<SparseSM::Worland::Boundary::Value>();
               bcOp.addRow<SparseSM::Worland::Boundary::D1>();
            }
            else if(bcId == 1)
            {
               bcOp.addRow<SparseSM::Worland::Boundary::Value>();
               bcOp.addRow<SparseSM::Worland::Boundary::D2>();
            }
            else
            {
               throw std::logic_error("Boundary conditions for Temperature not implemented");
            }
         }
         else if(fieldId == PhysicalNames::Temperature::id() && compId == FieldComponents::Spectral::SCALAR)
         {
            if(bcId == 0)
            {
               bcOp.addRow<SparseSM::Worland::Boundary::Value>();
            }
            else if(bcId == 1)
            {
               bcOp.addRow<SparseSM::Worland::Boundary::D1>();
            }
            else
            {
               throw std::logic_error("Boundary conditions for Temperature not implemented");
            }
         }

         rModelMatrix.real() += bcOp.mat();
      }
   }

   void ModelBackend::galerkinStencil(SparseMatrix& mat, const SpectralFieldId& fId, const int matIdx, const Resolution& res, const std::vector<MHDFloat>& eigs, const bool makeSquare, const BcMap& bcs, const NonDimensional::NdMap& nds) const
   {
      assert(eigs.size() == 1);
      int l = eigs.at(0);

      auto fieldId = fId.first;
      auto compId = fId.second;

      auto bcId = bcs.find(PhysicalNames::Coordinator::tag(fieldId))->second;

      auto nN = res.counter().dimensions(Dimensions::Space::SPECTRAL, eigs.at(0))(0);

      auto a = Polynomial::Worland::WorlandBase::ALPHA_CHEBYSHEV;
      auto b = Polynomial::Worland::WorlandBase::DBETA_CHEBYSHEV;

      int s = 0;
      if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::TOR)
      {
         s = 1;
         if(bcId == 0)
         {
            SparseSM::Worland::Stencil::Value bc(nN, nN-1, a, b, l);
            mat = bc.mat(); 
         }
         else if(bcId == 1)
         {
            SparseSM::Worland::Stencil::R1D1DivR1 bc(nN, nN-1, a, b, l);
            mat = bc.mat(); 
         }
         else
         {
            throw std::logic_error("Boundary conditions for Temperature not implemented");
         }
      }
      else if(fieldId == PhysicalNames::Velocity::id() && compId == FieldComponents::Spectral::POL)
      {
         s = 2;
         if(bcId == 0)
         {
            SparseSM::Worland::Stencil::ValueD1 bc(nN, nN-2, a, b, l);
            mat = bc.mat(); 
         }
         else if(bcId == 1)
         {
            SparseSM::Worland::Stencil::ValueD2 bc(nN, nN-2, a, b, l);
            mat = bc.mat(); 
         }
         else
         {
            throw std::logic_error("Boundary conditions for Temperature not implemented");
         }
      }
      else if(fieldId == PhysicalNames::Temperature::id() && compId == FieldComponents::Spectral::SCALAR)
      {
         s = 1;
         if(bcId == 0)
         {
            SparseSM::Worland::Stencil::Value bc(nN, nN-1, a, b, l);
            mat = bc.mat(); 
         }
         else if(bcId == 1)
         {
            SparseSM::Worland::Stencil::D1 bc(nN, nN-1, a, b, l);
            mat = bc.mat(); 
         }
         else
         {
            throw std::logic_error("Boundary conditions for Temperature not implemented");
         }
      }

      if(makeSquare)
      {
         SparseSM::Worland::Id qId(nN-s, nN, a, b, l);
         mat = qId.mat()*mat;
      }
   }

   void ModelBackend::explicitBlock(DecoupledZSparse& mat, const SpectralFieldId& fId, const std::size_t opId,  const SpectralFieldId fieldId, const int matIdx, const Resolution& res, const std::vector<MHDFloat>& eigs, const BcMap& bcs, const NonDimensional::NdMap& nds) const
   {
      // Explicit linear operator
      if(opId == ModelOperator::ExplicitLinear::id())
      {
         // Nothing to be done
         throw std::logic_error("There are no explicit linear operators");
      }
      // Explicit nonlinear operator
      else if(opId == ModelOperator::ExplicitNonlinear::id())
      {
         throw std::logic_error("There are no explicit nonlinear operators");
      }
      // Explicit nextstep operator
      else if(opId == ModelOperator::ExplicitNextstep::id())
      {
         throw std::logic_error("There are no explicit nextstep operators");
      }
   }

}
}
}
}
}
}
