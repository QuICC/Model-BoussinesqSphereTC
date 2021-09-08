/**
 * @file ITCModel.cpp
 * @brief Source of the Boussinesq thermal convection in a sphere (Toroidal/Poloidal formulation)
 */

// Configuration includes
//

// System includes
//

// External includes
//

// Class include
//
#include "QuICC/Model/Boussinesq/Sphere/TC/ITCModel.hpp"

// Project includes
//
#include "QuICC/Model/Boussinesq/Sphere/TC/Transport.hpp"
#include "QuICC/Model/Boussinesq/Sphere/TC/Momentum.hpp"
#include "QuICC/Enums/FieldIds.hpp"
#include "QuICC/PhysicalNames/Temperature.hpp"
#include "QuICC/PhysicalNames/Velocity.hpp"
#include "QuICC/NonDimensional/Prandtl.hpp"
#include "QuICC/NonDimensional/Rayleigh.hpp"
#include "QuICC/Io/Variable/StateFileReader.hpp"
#include "QuICC/Io/Variable/StateFileWriter.hpp"
#include "QuICC/Io/Variable/VisualizationFileWriter.hpp"
#include "QuICC/Io/Variable/SphereNusseltWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarEnergyWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarLSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarMSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolEnergyWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolLSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolMSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolEnstrophyWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolEnstrophyLSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolEnstrophyMSpectrumWriter.hpp"
#include "QuICC/Generator/States/RandomScalarState.hpp"
#include "QuICC/Generator/States/RandomVectorState.hpp"
#include "QuICC/Generator/States/SphereExactStateIds.hpp"
#include "QuICC/Generator/States/SphereExactScalarState.hpp"
#include "QuICC/Generator/States/SphereExactVectorState.hpp"
#include "QuICC/Generator/Visualizers/ScalarFieldVisualizer.hpp"
#include "QuICC/Generator/Visualizers/VectorFieldVisualizer.hpp"
#include "QuICC/SpectralKernels/MakeRandom.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

   VectorFormulation::Id ITCModel::SchemeFormulation()
   {
      return VectorFormulation::TORPOL;
   }

   void ITCModel::addEquations(SharedSimulation spSim)
   {
     // Add transport equation
      spSim->addEquation<Equations::Boussinesq::Sphere::TC::Transport>(this->spBackend());

      // Add Navier-Stokes equation
      spSim->addEquation<Equations::Boussinesq::Sphere::TC::Momentum>(this->spBackend());
   }

   void ITCModel::addStates(SharedStateGenerator spGen)
   {
      // Shared pointer to equation
      Equations::SharedSphereExactScalarState spScalar;
      Equations::SharedSphereExactVectorState spVector;

      Spectral::Kernel::Complex3DMapType tSH;
      std::pair<Spectral::Kernel::Complex3DMapType::iterator,bool> ptSH;

      // Add temperature initial state generator
      spScalar = spGen->addEquation<Equations::SphereExactScalarState>(this->spBackend());
      spScalar->setIdentity(PhysicalNames::Temperature::id());
      switch(3)
      {
         case 0:
            {
               spScalar->setPhysicalNoise(1e-15);
            }
            break;

         case 1:
            {
               spScalar->setPhysicalConstant(1.0);
            }
            break;

         case 2:
            {
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(3,3), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0,2.0)));
               spScalar->setSpectralModes(tSH);
            }
            break;

         case 3:
            {
               auto spKernel = std::make_shared<Spectral::Kernel::MakeRandom>(spGen->ss().has(SpatialScheme::Feature::ComplexSpectrum));
               std::vector<MHDFloat> ratios = {1e4, 1e4, 1e4};
               spKernel->setRatio(ratios);
               spKernel->init(-1e-2, 1e-2);
               spScalar->setSrcKernel(spKernel);
            }
            break;
      }

      // Add velocity initial state generator
      spVector = spGen->addEquation<Equations::SphereExactVectorState>(this->spBackend());
      spVector->setIdentity(PhysicalNames::Velocity::id());
      switch(3)
      {
         // Toroidal only
         case 0:
            {
               // Toroidal
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(1,1), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
               spVector->setSpectralModes(FieldComponents::Spectral::TOR, tSH);
               // Poloidal
               tSH.clear();
               spVector->setSpectralModes(FieldComponents::Spectral::POL, tSH);
            }
            break;

         // Poloidal only
         case 1:
            {
               // Toroidal
               tSH.clear();
               spVector->setSpectralModes(FieldComponents::Spectral::TOR, tSH);
               // Poloidal
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(2,0), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
               spVector->setSpectralModes(FieldComponents::Spectral::POL, tSH);
            }
            break;

         // Toroidal & Poloidal
         case 2:
            {
               // Toroidal
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(1,1), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
               spVector->setSpectralModes(FieldComponents::Spectral::TOR, tSH);
               // Poloidal
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(2,0), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
               spVector->setSpectralModes(FieldComponents::Spectral::POL, tSH);
            }
            break;

         case 3:
            {
               auto spKernel = std::make_shared<Spectral::Kernel::MakeRandom>(spGen->ss().has(SpatialScheme::Feature::ComplexSpectrum));
               std::vector<MHDFloat> ratios = {1e4, 1e4, 1e4};
               spKernel->setRatio(ratios);
               spKernel->init(-1e-2, 1e-2);
               spVector->setSrcKernel(FieldComponents::Spectral::TOR, spKernel);
               spVector->setSrcKernel(FieldComponents::Spectral::POL, spKernel);
            }
            break;
      }

      // Add output file
      auto spOut = std::make_shared<Io::Variable::StateFileWriter>(spGen->ss().tag(), spGen->ss().has(SpatialScheme::Feature::RegularSpectrum));
      spOut->expect(PhysicalNames::Temperature::id());
      spOut->expect(PhysicalNames::Velocity::id());
      spGen->addHdf5OutputFile(spOut);
   }

   void ITCModel::addVisualizers(SharedVisualizationGenerator spVis)
   {
      // Shared pointer to basic field visualizer
      Equations::SharedScalarFieldVisualizer spScalar;
      Equations::SharedVectorFieldVisualizer spVector;

      // Add temperature field visualization
      spScalar = spVis->addEquation<Equations::ScalarFieldVisualizer>(this->spBackend());
      spScalar->setFields(true, true);
      spScalar->setIdentity(PhysicalNames::Temperature::id());

      // Add velocity field visualization
      spVector = spVis->addEquation<Equations::VectorFieldVisualizer>(this->spBackend());
      spVector->setFields(true, false, true);
      spVector->setIdentity(PhysicalNames::Velocity::id());

      // Add output file
      auto spOut = std::make_shared<Io::Variable::VisualizationFileWriter>(spVis->ss().tag());
      spOut->expect(PhysicalNames::Temperature::id());
      spOut->expect(PhysicalNames::Velocity::id());
      spVis->addHdf5OutputFile(spOut);
   }

   void ITCModel::addAsciiOutputFiles(SharedSimulation spSim)
   {
      // Create Nusselt writer
      auto spNusselt = std::make_shared<Io::Variable::SphereNusseltWriter>("", spSim->ss().tag());
      spNusselt->expect(PhysicalNames::Temperature::id());
      spSim->addAsciiOutputFile(spNusselt);

      // Create temperature energy writer
      auto spTemp = std::make_shared<Io::Variable::SphereScalarEnergyWriter>("temperature", spSim->ss().tag());
      spTemp->expect(PhysicalNames::Temperature::id());
      spSim->addAsciiOutputFile(spTemp);

#if 0
      // Create temperature L energy spectrum writer
      auto spTempL = std::make_shared<Io::Variable::SphereScalarLSpectrumWriter>("temperature", spSim->ss().tag());
      spTempL->expect(PhysicalNames::Temperature::id());
      //spTempL->numberOutput();
      //spTempL->onlyEvery(5);
      spSim->addAsciiOutputFile(spTempL);

      // Create temperature M energy spectrum writer
      auto spTempM = std::make_shared<Io::Variable::SphereScalarMSpectrumWriter>("temperature", spSim->ss().tag());
      spTempM->expect(PhysicalNames::Temperature::id());
      //spTempM->numberOutput();
      //spTempM->onlyEvery(5);
      spSim->addAsciiOutputFile(spTempM);
#endif

      // Create kinetic energy writer
      auto spKinetic = std::make_shared<Io::Variable::SphereTorPolEnergyWriter>("kinetic", spSim->ss().tag());
      spKinetic->expect(PhysicalNames::Velocity::id());
      spSim->addAsciiOutputFile(spKinetic);

#if 1
      // Create kinetic L energy spectrum writer
      auto spKineticL = std::make_shared<Io::Variable::SphereTorPolLSpectrumWriter>("kinetic", spSim->ss().tag());
      spKineticL->expect(PhysicalNames::Velocity::id());
      spKineticL->numberOutput();
      spKineticL->onlyEvery(5);
      spSim->addAsciiOutputFile(spKineticL);

      // Create kinetic M energy spectrum writer
      auto spKineticM = std::make_shared<Io::Variable::SphereTorPolMSpectrumWriter>("kinetic", spSim->ss().tag());
      spKineticM->expect(PhysicalNames::Velocity::id());
      spKineticM->numberOutput();
      spKineticM->onlyEvery(5);
      spSim->addAsciiOutputFile(spKineticM);
#endif

#if 0
      // Create enstrophy writer
      auto spEnstrophy = std::make_shared<Io::Variable::SphereTorPolEnstrophyWriter>("kinetic", spSim->ss().tag());
      spEnstrophy->expect(PhysicalNames::Velocity::id());
      spSim->addAsciiOutputFile(spEnstrophy);

      // Create enstrophy L spectrum writer
      auto spEnstrophyL = std::make_shared<Io::Variable::SphereTorPolEnstrophyLSpectrumWriter>("kinetic", spSim->ss().tag());
      spEnstrophyL->expect(PhysicalNames::Velocity::id());
      //spEnstrophyL->numberOutput();
      //spEnstrophyL->onlyEvery(5);
      spSim->addAsciiOutputFile(spEnstrophyL);

      // Create enstrophy M spectrum writer
      auto spEnstrophyM = std::make_shared<Io::Variable::SphereTorPolEnstrophyMSpectrumWriter>("kinetic", spSim->ss().tag());
      spEnstrophyM->expect(PhysicalNames::Velocity::id());
      //spEnstrophyM->numberOutput();
      //spEnstrophyM->onlyEvery(5);
      spSim->addAsciiOutputFile(spEnstrophyM);
#endif
   }

}
}
}
}
}
