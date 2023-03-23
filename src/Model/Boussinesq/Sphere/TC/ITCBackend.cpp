/**
 * @file ITCBackend.cpp
 * @brief Source of the interface for model backend
 */

// System includes
//
#include <stdexcept>

// Project includes
//
#include "QuICC/Model/Boussinesq/Sphere/TC/ITCBackend.hpp"
#include "QuICC/ModelOperator/Time.hpp"
#include "QuICC/ModelOperator/ImplicitLinear.hpp"
#include "QuICC/ModelOperator/ExplicitLinear.hpp"
#include "QuICC/ModelOperator/ExplicitNonlinear.hpp"
#include "QuICC/ModelOperator/ExplicitNextstep.hpp"
#include "QuICC/ModelOperator/Stencil.hpp"
#include "QuICC/ModelOperator/Boundary.hpp"
#include "QuICC/ModelOperator/SplitImplicitLinear.hpp"
#include "QuICC/ModelOperator/SplitBoundary.hpp"
#include "QuICC/ModelOperatorBoundary/FieldToRhs.hpp"
#include "QuICC/ModelOperatorBoundary/SolverHasBc.hpp"
#include "QuICC/ModelOperatorBoundary/SolverNoTau.hpp"
#include "QuICC/ModelOperatorBoundary/Stencil.hpp"
#include "QuICC/Enums/FieldIds.hpp"
#include "QuICC/Bc/Name/FixedTemperature.hpp"
#include "QuICC/Bc/Name/FixedFlux.hpp"
#include "QuICC/Bc/Name/StressFree.hpp"
#include "QuICC/Bc/Name/NoSlip.hpp"
#include "QuICC/PhysicalNames/Velocity.hpp"
#include "QuICC/PhysicalNames/Temperature.hpp"
#include "QuICC/NonDimensional/Prandtl.hpp"
#include "QuICC/NonDimensional/Rayleigh.hpp"
#include "QuICC/NonDimensional/CflInertial.hpp"
#include "QuICC/Tools/IdToHuman.hpp"
#include "QuICC/Resolutions/Tools/IndexCounter.hpp"
#include "QuICC/Polynomial/Worland/WorlandBase.hpp"
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
#include "QuICC/SparseSM/Worland/Id.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

   ITCBackend::ITCBackend()
      : IModelBackend(), mUseGalerkin(false), mUseSplitEquation(false)
   {
   }

   std::vector<std::string> ITCBackend::fieldNames() const
   {
      std::vector<std::string> names = {
         PhysicalNames::Velocity().tag(),
         PhysicalNames::Temperature().tag()
      };

      return names;
   }

   std::vector<std::string> ITCBackend::paramNames() const
   {
      std::vector<std::string> names = {
         NonDimensional::Prandtl().tag(),
         NonDimensional::Rayleigh().tag(),
      };

      return names;
   }

   std::vector<bool> ITCBackend::isPeriodicBox() const
   {
      std::vector<bool> periodic = {false, false, false};

      return periodic;
   }

   bool ITCBackend::useGalerkin() const
   {
      return this->mUseGalerkin;
   }

   void ITCBackend::enableGalerkin(const bool flag)
   {
      this->mUseGalerkin = flag;
   }

   bool ITCBackend::useSplitEquation() const
   {
      return this->mUseSplitEquation;
   }

   void ITCBackend::enableSplitEquation(const bool tag)
   {
      this->mUseSplitEquation = tag;
   }

   std::map<std::string,MHDFloat> ITCBackend::automaticParameters(const std::map<std::string,MHDFloat>& cfg) const
   {
      std::map<std::string,MHDFloat> params;

      return params;
   }

   int ITCBackend::nBc(const SpectralFieldId& fId) const
   {
      int nBc = 0;

      if(fId == std::make_pair(PhysicalNames::Velocity::id(), FieldComponents::Spectral::TOR) ||
            fId == std::make_pair(PhysicalNames::Temperature::id(), FieldComponents::Spectral::SCALAR))
      {
         nBc = 1;
      }
      else if(fId == std::make_pair(PhysicalNames::Velocity::id(), FieldComponents::Spectral::POL))
      {
         nBc = 2;
      }
      else
      {
         nBc = 0;
      }

      return nBc;
   }

   void ITCBackend::blockInfo(int& tN, int& gN, ArrayI& shift, int& rhs, const SpectralFieldId& fId, const Resolution& res, const MHDFloat l, const BcMap& bcs) const
   {
      auto nN = res.counter().dimensions(Dimensions::Space::SPECTRAL, l)(0);
      tN = nN;

      int shiftR = this->nBc(fId);
      if(this->useGalerkin())
      {
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

   void ITCBackend::applyTau(SparseMatrix& mat, const SpectralFieldId& rowId, const SpectralFieldId& colId, const int l, const Resolution& res, const BcMap& bcs, const NonDimensional::NdMap& nds, const bool isSplitOperator) const
   {
      auto nN = res.counter().dimensions(Dimensions::Space::SPECTRAL, l)(0);

      auto a = Polynomial::Worland::WorlandBase::ALPHA_CHEBYSHEV;
      auto b = Polynomial::Worland::WorlandBase::DBETA_CHEBYSHEV;

      auto bcId = bcs.find(rowId.first)->second;

      SparseSM::Worland::Boundary::Operator bcOp(nN, nN, a, b, l);

      if(rowId == std::make_pair(PhysicalNames::Velocity::id(), FieldComponents::Spectral::TOR) && rowId == colId)
      {
         if(bcId == Bc::Name::NoSlip::id())
         {
            bcOp.addRow<SparseSM::Worland::Boundary::Value>();
         }
         else if(bcId == Bc::Name::StressFree::id())
         {
            bcOp.addRow<SparseSM::Worland::Boundary::R1D1DivR1>();
         }
         else
         {
            throw std::logic_error("Boundary conditions for Velocity Toroidal component not implemented");
         }
      }
      else if(rowId == std::make_pair(PhysicalNames::Velocity::id(), FieldComponents::Spectral::POL) && rowId == colId)
      {
         if(this->useSplitEquation())
         {
            if(isSplitOperator)
            {
               bcOp.addRow<SparseSM::Worland::Boundary::Value>();
            }
            else if(bcId == Bc::Name::NoSlip::id())
            {
               bcOp.addRow<SparseSM::Worland::Boundary::D1>();
            }
            else if(bcId == Bc::Name::StressFree::id())
            {
               bcOp.addRow<SparseSM::Worland::Boundary::D2>();
            }
            else
            {
               throw std::logic_error("Boundary conditions for Velocity Poloidal component not implemented");
            }
         }
         else
         {
            if(bcId == Bc::Name::NoSlip::id())
            {
               bcOp.addRow<SparseSM::Worland::Boundary::Value>();
               bcOp.addRow<SparseSM::Worland::Boundary::D1>();
            }
            else if(bcId == Bc::Name::StressFree::id())
            {
               bcOp.addRow<SparseSM::Worland::Boundary::Value>();
               bcOp.addRow<SparseSM::Worland::Boundary::D2>();
            }
            else
            {
               throw std::logic_error("Boundary conditions for Velocity Poloidal component not implemented");
            }
         }
      }
      else if(rowId == std::make_pair(PhysicalNames::Temperature::id(), FieldComponents::Spectral::SCALAR) && rowId == colId)
      {
         if(bcId == Bc::Name::FixedTemperature::id())
         {
            bcOp.addRow<SparseSM::Worland::Boundary::Value>();
         }
         else if(bcId == Bc::Name::FixedFlux::id())
         {
            bcOp.addRow<SparseSM::Worland::Boundary::D1>();
         }
         else
         {
            throw std::logic_error("Boundary conditions for Temperature not implemented (" + std::to_string(bcId) + ")");
         }
      }

      mat.real() += bcOp.mat();
   }

   void ITCBackend::stencil(SparseMatrix& mat, const SpectralFieldId& fieldId, const int l, const Resolution& res, const bool makeSquare, const BcMap& bcs, const NonDimensional::NdMap& nds) const
   {
      auto nN = res.counter().dimensions(Dimensions::Space::SPECTRAL, l)(0);

      auto a = Polynomial::Worland::WorlandBase::ALPHA_CHEBYSHEV;
      auto b = Polynomial::Worland::WorlandBase::DBETA_CHEBYSHEV;

      auto bcId = bcs.find(fieldId.first)->second;

      int s = this->nBc(fieldId);
      if(fieldId == std::make_pair(PhysicalNames::Velocity::id(), FieldComponents::Spectral::TOR))
      {
         if(bcId == Bc::Name::NoSlip::id())
         {
            SparseSM::Worland::Stencil::Value bc(nN, nN-s, a, b, l);
            mat = bc.mat();
         }
         else if(bcId == Bc::Name::StressFree::id())
         {
            SparseSM::Worland::Stencil::R1D1DivR1 bc(nN, nN-s, a, b, l);
            mat = bc.mat();
         }
         else
         {
            throw std::logic_error("Galerkin boundary conditions for Velocity Toroidal component not implemented");
         }
      }
      else if(fieldId == std::make_pair(PhysicalNames::Velocity::id(), FieldComponents::Spectral::POL))
      {
         if(bcId == Bc::Name::NoSlip::id())
         {
            SparseSM::Worland::Stencil::ValueD1 bc(nN, nN-s, a, b, l);
            mat = bc.mat();
         }
         else if(bcId == Bc::Name::StressFree::id())
         {
            SparseSM::Worland::Stencil::ValueD2 bc(nN, nN-s, a, b, l);
            mat = bc.mat();
         }
         else
         {
            throw std::logic_error("Galerin boundary conditions for Velocity Poloidal component not implemented");
         }
      }
      else if(fieldId == std::make_pair(PhysicalNames::Temperature::id(), FieldComponents::Spectral::SCALAR))
      {
         if(bcId == Bc::Name::FixedTemperature::id())
         {
            SparseSM::Worland::Stencil::Value bc(nN, nN-s, a, b, l);
            mat = bc.mat();
         }
         else if(bcId == Bc::Name::FixedFlux::id())
         {
            SparseSM::Worland::Stencil::D1 bc(nN, nN-s, a, b, l);
            mat = bc.mat();
         }
         else
         {
            throw std::logic_error("Galerkin boundary conditions for Temperature not implemented");
         }
      }

      if(makeSquare)
      {
         SparseSM::Worland::Id qId(nN-s, nN, a, b, l);
         mat = qId.mat()*mat;
      }
   }

   void ITCBackend::applyGalerkinStencil(SparseMatrix& mat, const SpectralFieldId& rowId, const SpectralFieldId& colId, const int l, const Resolution& res, const BcMap& bcs, const NonDimensional::NdMap& nds) const
   {
      auto nN = res.counter().dimensions(Dimensions::Space::SPECTRAL, l)(0);

      auto a = Polynomial::Worland::WorlandBase::ALPHA_CHEBYSHEV;
      auto b = Polynomial::Worland::WorlandBase::DBETA_CHEBYSHEV;

      auto S = mat;
      this->stencil(S, colId, l, res, false, bcs, nds);

      auto s = this->nBc(rowId);
      SparseSM::Worland::Id qId(nN-s, nN, a, b, l, 0, s);
      mat = qId.mat()*(mat*S);
   }

} // TC
} // Sphere
} // Boussinesq
} // Model
} // QuICC
